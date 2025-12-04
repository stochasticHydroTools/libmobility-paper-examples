import numpy as np
from math import floor
import subprocess
import logging
import os

# Module-level logger
logger = logging.getLogger(__name__)

# this code requires the C++ code linked below to be compiled within the directory C++
# code: https://torquatocpanel.deptcpanel.princeton.edu/links-and-codes/sphere-packings-registration/
# article: M. Skoge, A. Donev, F. H. Stillinger and S. Torquato, Packing Hyperspheres in High-Dimensional Euclidean Spaces, Physical Review E 74, 041127 (2006).

self_path = os.path.dirname(os.path.abspath(__file__))
out_dir = self_path + "/timing_configs/"


def generate_configs():

    N = 1000
    L_vals = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0])
    a = 0.05

    phi_vals = (4 / 3) * np.pi * N * a**3 / (L_vals**3)
    print("phi vals:", phi_vals)
    for j, phi in enumerate(phi_vals):

        out_file = out_dir + f"sphere_pack_{L_vals[j]}.txt"
        write_conf(phi, N)

        subprocess.run([self_path + "/C++/a.out", self_path + "/C++/input"])

        with open("temp.dat") as f:
            lines = f.readlines()
            radius = float(lines[3].strip())
            N = int(lines[2].strip())
            lines = lines[6:]
            dat = np.zeros((N, 3))
            for i, l in enumerate(lines):
                x, y, z = map(float, l.strip().split())
                dat[i] = np.array([x, y, z]) * L_vals[j]

        with open(out_file, "w") as f:
            # scale output radius since the C++ code uses unit box
            f.write(f"# N: {N}, a: {0.5*radius*L_vals[j]}\n")
            np.savetxt(f, dat, fmt="%.6f")

        subprocess.run(["rm", "junk.dat"])


def write_conf(phi, N):
    logger.info("generating conf for %d particles", N)
    with open(self_path + "/C++/input", "w") as f:
        f.write("int eventspercycle = 20;\n")
        f.write(f"int N = {N};\n")
        f.write("double initialpf = 0.01;\n")
        f.write(f"double maxpf = {phi};\n")
        f.write("double temp = 0.2;\n")
        f.write("double growthrate = 0.001;\n")
        f.write("double maxpressure = 100.0;\n")
        f.write("char* readfile = new\n")
        f.write(f"char* writefile = temp.dat\n")
        f.write("char* datafile = junk.dat\n")


if __name__ == "__main__":
    generate_configs()
