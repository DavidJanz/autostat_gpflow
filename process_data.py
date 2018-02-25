import numpy as np

with open("data/co2_raw.txt") as fhandle:
    entries = np.array([l.split()[2:4] for l in fhandle.readlines()
                        if "#" not in l], dtype=float)
    entries = entries[entries[:, 1] > 0]
    np.savez("data/co2.npz", **{'x': entries[:, 0], 'y': entries[:, 1]})
