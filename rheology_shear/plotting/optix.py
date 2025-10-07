from functools import partial
import numpy as np
from numba import njit, prange
from plotoptix import NpOptiX as OptiX  # headless renderer
from plotoptix.materials import m_plastic
import matplotlib.pyplot as plt
import matplotlib.colors
from threading import Event
from PIL import Image
import logging
import os
import json

ACCUM_DONE = Event()
logger = logging.getLogger(__name__)


def init(rt):

    max_frames = 128
    rt.set_param(
        min_accumulation_step=4,
        max_accumulation_frames=max_frames,
    )
    rt.set_uint("path_seg_range", 4, 16)
    rt.set_float("scene_epsilon", 0.001)

    rt.set_background(0)
    rt.set_ambient(1.15)

    rt.setup_material("plastic", m_plastic)

    exposure = 1.1
    gamma = 2.2
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_igamma", 1 / gamma)
    rt.set_uint("denoiser_start", max_frames)
    rt.set_float("denoiser_blend", 3.0)
    rt.add_postproc("Denoiser")

    scale_fact = 1.0

    current_dir = os.path.abspath(os.path.dirname(__file__))
    fname = os.path.join(current_dir, "stress_per_blob.csv")
    dat = np.loadtxt(fname, delimiter=",", skiprows=2)
    blobs = dat[:, 0:3]
    blobs = np.reshape(blobs, (-1, 3))
    S_blob = dat[:, 3]

    plot_params = json.load(open(os.path.join(current_dir, "plot_params.json"), "r"))
    a = plot_params["a"]
    blobs_per_body = plot_params["n_per_body"]

    S_shift = S_blob - np.min(S_blob) + 1e-5
    S_log = np.log10(S_shift)
    S_plot = S_log

    shift = 0.3  # shifting like this puts a nice cut on the corner
    blobs = blobs + np.array([shift, shift, shift])
    blobs = periodize_r_vecs(blobs, np.array([1.0, 1.0, 1.0]), blobs.shape[0])
    blobs[:, 2] -= 0.25  # flatten in z for better visualization
    logger.info(
        "max blob coords %s min blob coords %s",
        np.max(blobs, axis=0),
        np.min(blobs, axis=0),
    )
    logger.info("max stress %s min stress %s", np.max(S_plot), np.min(S_plot))

    cmap = plt.get_cmap("magma")
    norm = matplotlib.colors.Normalize(vmin=S_plot.min(), vmax=S_plot.max())
    colors = np.zeros_like(blobs)
    for i, cval in enumerate(S_plot):
        colors[i, :] = cmap(norm(cval))[0:3]
    N = blobs.shape[0]

    blobs *= scale_fact

    rad_scale = 1
    rt.set_data(
        "blobs",
        mat="plastic",
        pos=blobs,
        r=rad_scale * a,
        geom="ParticleSetConstSize",
        c=colors,
    )
    rt.setup_camera(
        "cam1",
        cam_type="DoF",
        eye=[1.18265343, 1.22631025, 1.190259],
        target=[1.0, 1.0, 0.7],
        up=[-0.371450275, -0.4319413, 0.8218586],
        aperture_radius=0.001,
        fov=105.0,
        focal_scale=0.4,
    )


@njit(parallel=True, fastmath=True)
def periodize_r_vecs(r_vecs_np, L, Nb):
    r_vecs = np.copy(r_vecs_np)
    for k in prange(Nb):
        for i in range(3):
            if L[i] > 0:
                while r_vecs[k, i] < 0:
                    r_vecs[k, i] += L[i]
                while r_vecs[k, i] > L[i]:
                    r_vecs[k, i] -= L[i]
    return r_vecs


def save_image(rt, fname):
    """Headless save: fetch OptiX framebuffer and write PNG/JPG with Pillow."""
    logger.debug("camera: %s", rt.get_camera("cam1"))

    img = rt.get_rt_output()
    if img is None:
        raise RuntimeError("rt.get_rt_output() returned None (no framebuffer).")
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected image shape from OptiX: {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype is not np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    Image.fromarray(img, mode="RGB").save(fname)
    logger.info("Saved image to %s", os.path.abspath(fname))
    ACCUM_DONE.set()
    logger.info("rt completed!")


def main(out_fname="optix_scene.png"):
    initialize = partial(init)
    write_image_to_file = partial(save_image, fname=out_fname)

    optix = OptiX(
        on_rt_accum_done=write_image_to_file,
        on_initialization=initialize,
        start_now=True,
        width=1024,
        height=1024,
    )
    ACCUM_DONE.wait()
    optix.close()
    logger.info("done")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
