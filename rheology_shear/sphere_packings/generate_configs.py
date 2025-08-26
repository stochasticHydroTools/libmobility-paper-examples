import numpy as np
from math import floor
import subprocess

# this code requires the C++ code linked below to be compiled within the directory C++
# code: https://torquatocpanel.deptcpanel.princeton.edu/links-and-codes/sphere-packings-registration/


def main():
    phi_vals = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    for phi in phi_vals:

        out_file = f"configs/sphere_pack_{phi:0.3g}.txt"
        write_conf(phi)

        subprocess.run(["./C++/a.out", "./C++/input"])

        with open("temp.dat") as f:
            lines = f.readlines()
            radius = float(lines[3].strip())
            N = int(lines[2].strip())
            lines = lines[6:]
            dat = np.zeros((N, 3))
            for i, l in enumerate(lines):
                x, y, z = map(float, l.strip().split())
                dat[i] = [x, y, z]

        with open(out_file, "w") as f:
            f.write(f"# N: {N}, a: {radius}\n")
            np.savetxt(f, dat, fmt="%.6f")

        subprocess.run(["rm", "junk.dat"])


def write_conf(phi):
    L = np.array([1.0, 1.0, 1.0])
    a = 0.1

    N = floor(phi * L[0] * L[1] * L[2] / ((4 / 3) * np.pi * a**3))
    print(f"generating conf for {N} particles")
    with open("C++/input", "w") as f:
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
    main()
