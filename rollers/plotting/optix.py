from functools import partial
import os
import numpy as np
import logging
from PIL import Image

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from plotoptix.materials import m_plastic

from threading import Event

event = Event()


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
    event.set()
    logger.info("rt completed!")


def init(rt, hg, t_star):

    white = np.array([248, 248, 255]) / 255.0

    rt.set_param(
        min_accumulation_step=16, max_accumulation_frames=512, light_shading="Hard"
    )
    rt.set_uint("path_seg_range", 30, 50)
    rt.set_float("scene_epsilon", 0.04)

    rt.set_background(0)
    rt.set_ambient(1.15)

    rt.setup_material("plastic", m_plastic)

    exposure = 0.65
    gamma = 2.2
    rt.set_float("tonemap_exposure", exposure)
    rt.set_float("tonemap_igamma", 1 / gamma)
    rt.set_float("denoiser_blend", 3.0)
    rt.add_postproc("Denoiser")

    scale_fact = 0.2

    a = 0.656 * scale_fact
    Lx = 1500 * a
    Ly = 5400 * a

    det_color = np.array([100, 149, 237]) / 255.0
    stoch_color = np.array([255, 119, 255]) / 255.0

    parent = os.path.dirname(os.path.abspath(__file__))
    output_dir = parent + "/../output_data/"
    run = 0
    det_fname = output_dir + f"hg{hg}/deterministic/{run}/t_star/pos_{t_star}.csv"
    stoch_fname = output_dir + f"hg{hg}/stochastic/{run}/t_star/pos_{t_star}.csv"

    # this makes the bottom floor
    vertices, triangles = make_rectangle(0, 0, Lx, Ly)
    rt.set_mesh("bottom_wall", vertices, triangles, c=1.4 * white, mat="diffuse")

    det_pos = np.loadtxt(det_fname, delimiter=",")
    stoch_pos = np.loadtxt(stoch_fname, delimiter=",")
    det_pos = np.reshape(det_pos, (-1, 3))
    stoch_pos = np.reshape(stoch_pos, (-1, 3))

    det_pos *= scale_fact
    stoch_pos *= scale_fact

    det_pos[:, 0] += 700 * a

    y_diff = (
        5400 - 4915
    ) * a  # how much larger the bottom wall is in the visualization
    det_pos[:, 1] += 0.5 * y_diff
    stoch_pos[:, 1] += 0.5 * y_diff

    rad_scale = 1
    rad_scale = 1
    rt.set_data(
        "particles_det",
        mat="plastic",
        pos=det_pos,
        r=rad_scale * a,
        geom="ParticleSetConstSize",
        c=det_color,
    )
    rt.set_data(
        "particles_stoch",
        mat="plastic",
        pos=stoch_pos,
        r=rad_scale * a,
        geom="ParticleSetConstSize",
        c=stoch_color,
    )

    rt.setup_camera(
        "cam1",
        cam_type="DoF",
        eye=[Lx / 2, Ly / 2, 130.0],
        up=[1.0, 0.0, 0.2],
        aperture_radius=0.001,
        fov=95.0,
        focal_scale=100.0,
    )

    # this makes a rectangle for a scale bar
    # vertices, triangles = make_rectangle(-100 * a, 500 * a, 20 * a, 1000 * a)
    # rt.set_mesh(
    #     "det_rectangle",
    #     vertices,
    #     triangles,
    #     c=plat,
    #     mat="plastic",
    # )


def main(hg=15, t_star="t_star"):

    parent = os.path.dirname(os.path.abspath(__file__))
    out_fname = parent + f"/plots/roller_hg{hg}_{t_star}.png"
    initialize = partial(init, hg=hg, t_star=t_star)
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
        width=2048,
        height=1024,
    )
    if headless:
        event.wait()
        logger.info("done")
        optix.close()


def make_rectangle(x0, y0, lx, ly, z=0):
    vertices = np.array(
        [[x0, y0, z], [x0 + lx, y0, z], [x0, y0 + ly, z], [x0 + lx, y0 + ly, z]]
    )
    triangles = np.array([[0, 1, 2], [1, 2, 3]])
    """
    Create a rectangle in 3D space.
    """
    return vertices, triangles


if __name__ == "__main__":
    main()
