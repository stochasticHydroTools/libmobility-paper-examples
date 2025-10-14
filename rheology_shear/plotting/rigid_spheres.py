from functools import partial
import numpy as np

from plotoptix.materials import m_plastic
import logging
import os
from threading import Event

logger = logging.getLogger(__name__)
ACCUM_DONE = Event()  # <-- signal when image is written


def init(rt):

    max_frames = 256
    rt.set_param(
        min_accumulation_step=8,
        max_accumulation_frames=max_frames,
    )

    rt.set_background(0)
    rt.set_ambient(0.5)

    rt.setup_material("plastic", m_plastic)

    exposure = 1.1
    gamma = 2.2
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_igamma", 1 / gamma)
    rt.set_uint("denoiser_start", 32)
    rt.set_float("denoiser_blend", 0.5)
    rt.add_postproc("Denoiser")

    colors = np.array(
        [
            [0.4767, 0.7453, 0.8163],
            [0.3433, 0.6134, 0.7656],
            [0.2593, 0.4796, 0.7186],
            [0.2458, 0.3399, 0.6394],
            [0.2195, 0.2214, 0.4564],
            [0.1298, 0.1258, 0.2548],
        ]
    )
    parent = os.path.abspath(os.path.dirname(__file__))
    struct_dir = os.path.join(parent, "..", "structures")
    N_vals = [12, 42, 162, 642, 2562]
    a_vals = np.zeros_like(N_vals, dtype=float)

    blobs = np.empty((np.sum(N_vals), 3))
    for i, N in enumerate(N_vals):
        fname = os.path.join(struct_dir, f"shell_N_{N}.csv")
        if not os.path.exists(fname):
            logger.error("Structure file not found: %s", fname)
            raise FileNotFoundError(f"Structure file not found: {fname}")
        struct_params, cfg = load_cfg(fname)
        a_vals[i] = 0.5 * struct_params["sep"]
        start = sum(N_vals[0:i])
        end = start + N
        blobs[start:end, :] = cfg

    positions = np.array(
        [
            [1.5, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [0.0, 0.0, -3.0],
            [3.0, 0.0, -3.0],
            [6.0, 0.0, -3.0],
        ]
    )

    for i in range(len(N_vals)):
        start = sum(N_vals[0:i])
        end = sum(N_vals[0 : i + 1])
        shift_blobs = blobs[start:end, :] + positions[i]
        logger.info("blob center %s", np.mean(shift_blobs, axis=0))
        logger.info("blob radius %s", a_vals[i])
        rt.set_data(
            f"blobs_{i}",
            mat="plastic",
            pos=shift_blobs,
            r=a_vals[i],
            geom="ParticleSetConstSize",
            c=colors[i],
        )

    for i in range(5):
        rt.setup_spherical_light(
            f"light_{i}",
            pos=positions[i] + np.array([-1.0, -2.0, 0]),
            color=30 * np.sqrt(i),
            radius=0.2,
            in_geometry=False,
        )

    rt.setup_camera(
        "cam1",
        cam_type="DoF",
        eye=[3, -5, -1],
        target=[3, 0, -1],
        up=[0, 0, 1],
        aperture_radius=0.001,
        fov=80.0,
        focal_scale=0.4,
    )


def save_image(rt, fname):
    """Headless save: fetch OptiX framebuffer and write PNG/JPG with Pillow."""
    from PIL import Image  # local import to keep dependencies minimal

    logger.debug("camera: %s", rt.get_camera("cam1"))

    img = rt.get_rt_output()  # expected uint8 array, shape (H, W, 4) or (H, W, 3)
    if img is None:
        raise RuntimeError("rt.get_rt_output() returned None (no framebuffer).")
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected image shape from OptiX: {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    Image.fromarray(img, mode="RGB").save(fname)
    logger.info("Saved image to %s", os.path.abspath(fname))
    logger.info("rt completed!")
    ACCUM_DONE.set()  # signal completion


def load_cfg(file_name):
    with open(file_name, "r") as f:
        _ = f.readline()
        params = f.readline().strip().split(",")
        sep = float(params[0].split(" ")[1])
        N = int(params[1])
        rg = float(params[2])
        rh = int(params[3])
        cfg = np.loadtxt(f, delimiter=" ")
        params = {"sep": sep, "N": N, "Rg": rg, "Rh": rh}
    return params, cfg


def main(out_fname="rigid_spheres.png"):

    initialize = partial(init)
    write_image_to_file = partial(save_image, fname=out_fname)

    headless = True
    if headless:
        from plotoptix import NpOptiX as OptiX
    else:
        from plotoptix import TkOptiX as OptiX

    optix = OptiX(
        on_rt_accum_done=write_image_to_file,
        on_initialization=initialize,
        start_now=True,
        width=1800,
        height=1024,
    )
    if headless:
        ACCUM_DONE.wait()
        optix.close()
        logger.info("done")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
