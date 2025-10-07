"""
Surface plot.
"""

from functools import partial
import os
import sys
import numpy as np
import logging
from PIL import Image

# Module-level logger
logger = logging.getLogger(__name__)
import plotoptix.enums as enums
from plotoptix import NpOptiX as Optix
from plotoptix.utils import map_to_colors, simplex
from plotoptix.materials import m_eye_normal_cos
from plotoptix.materials import (
    m_clear_glass,
    m_matt_glass,
    m_diffuse,
    m_matt_diffuse,
    m_plastic,
    m_matt_plastic,
    m_mirror,
    m_thin_walled,
    m_transparent_diffuse,
)

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


def chebs(a, b, n):
    i = np.array(range(n))
    x = np.cos((2 * i + 1) * np.pi / (2 * n))
    return 3.0 * (b - a) * x + 3.0 * (b + a)


def init(rt: Optix, hg, t_star):

    yellow = np.array([255.0 / 255.0, 174.0 / 255.0, 26.0 / 255.0])
    teal = np.array([67, 179, 174]) / 255.0
    bbblue = np.array([176, 196, 222]) / 255.0
    turq = np.array([0, 206, 209]) / 255.0
    ly = np.array([250, 240, 190]) / 255.0
    deeppeach = np.array([255, 203, 164]) / 255.0
    bubs = np.array([231, 254, 255]) / 255.0
    pblue = np.array([176, 224, 230]) / 255.0
    thistle = np.array([216, 191, 216]) / 255.0
    khaki = np.array([240, 230, 140]) / 255.0
    blueb = np.array([182, 225, 242]) / 255.0
    steel = np.array([242, 243, 244]) / 255.0
    white = np.array([248, 248, 255]) / 255.0
    pink = np.array([255, 0, 144]) / 255.0
    silver = np.array([191, 193, 194]) / 255.0
    silver2 = np.array([169, 169, 172]) / 255.0
    dark_pink = np.array([139, 0, 139]) / 255.0
    orange = np.array([255, 211, 0]) / 255.0
    chart = np.array([127, 255, 0]) / 255.0
    mint = np.array([0, 250, 154]) / 255.0
    lime = np.array([50, 205, 50]) / 255.0
    canary = np.array([255, 239, 0]) / 255.0
    plat = np.array([229, 228, 226]) / 255.0
    plat_cool = np.array([226, 228, 229]) / 255.0
    jet = np.array([53, 56, 57]) / 255.0

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
    # rt.set_float("denoiser_blend", 0.7)
    rt.set_float("denoiser_blend", 3.0)
    # rt.add_postproc("Gamma")  # apply gamma correction postprocessing stage, or
    rt.add_postproc("Denoiser")

    # rt.set_uint("path_seg_range", 15, 30)

    scale_fact = 0.2

    a = 0.656 * scale_fact
    Lx = 1500 * a
    Ly = 5400 * a

    det_color = np.array([100, 149, 237]) / 255.0
    stoc_color = np.array([255, 119, 255]) / 255.0

    grav_height = f"hg{hg}/"
    run = 0
    det_fname = grav_height + f"deterministic/output/pos_{t_star}_{run}.csv"
    stoch_fname = grav_height + f"stochastic/output/pos_{t_star}_{run}.csv"

    # this makes the bottom floor
    vertices, triangles = make_rectangle(0, 0, Lx, Ly)
    rt.set_mesh("bottom_wall", vertices, triangles, c=1.4 * white, mat="diffuse")

    det_pos = np.loadtxt(det_fname, delimiter=",")
    stoch_pos = np.loadtxt(stoch_fname, delimiter=",")
    det_pos = np.reshape(det_pos, (-1, 3))
    stoch_pos = np.reshape(stoch_pos, (-1, 3))
    N = det_pos.shape[0]

    det_pos *= scale_fact
    stoch_pos *= scale_fact

    # stoch_pos[:, 0] += 100 * a
    det_pos[:, 0] += 700 * a

    y_diff = (
        5400 - 4915
    ) * a  # how much larger the bottom wall is in the visualization
    det_pos[:, 1] += 0.5 * y_diff
    stoch_pos[:, 1] += 0.5 * y_diff

    rad_scale = 1
    for i in range(N):
        rt.set_data(
            "particles_det_" + str(i),
            mat="diffuse",
            pos=det_pos[i, :],
            r=rad_scale * a,
            c=det_color,
        )

        rt.set_data(
            "particles_stoch_" + str(i),
            mat="diffuse",
            pos=stoch_pos[i, :],
            r=rad_scale * a,
            c=stoc_color,
        )

    rt.setup_camera(
        "cam1",
        cam_type="DoF",  # comment out to use default, pinhole camera
        eye=[Lx / 2, 200 * a, 30],
        target=[Lx / 2, 10, 10],
    )

    # scale rectangle

    vertices, triangles = make_rectangle(-100 * a, 500 * a, 20 * a, 1000 * a)
    rt.set_mesh(
        "det_rectangle",
        vertices,
        triangles,
        c=plat,
        mat="plastic",
    )


def main(hg=61, t_star="2t_star", out_fname="TEMP.png"):
    # out_fname = f"img/roller_hg{hg}_{t_star}.png"
    out_fname = "TEMP.png"

    initialize = partial(init, hg=hg, t_star=t_star)
    write_image_to_file = partial(save_image, fname=out_fname)

    optix = Optix(
        on_rt_accum_done=write_image_to_file,
        on_initialization=initialize,
        start_now=True,
        width=4140,
        height=1024,
    )
    event.wait()
    logger.info("done")


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
