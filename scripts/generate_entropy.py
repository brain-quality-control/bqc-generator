import json
import numpy as np


x = np.load("std_entropy.npz")
i = np.argsort(x["std"])
_std = x["std"][i]
_subjects = x["subjects"][i]

with open("static/entropy.json", "w") as f:
    json.dump({"std": _std.tolist(), "subjects": _subjects.tolist()}, f)
