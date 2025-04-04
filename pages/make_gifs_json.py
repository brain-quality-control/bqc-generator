import json
import glob
import os

# Get all the gifs
gifs = glob.glob("static/gifs/gif/*.gif")

data = []
for gif in gifs:
    # Get the name of the gif
    name = os.path.basename(gif).split(".")[0]
    # Get the pngs sorted by index
    pngs = sorted(
        glob.glob(f"static/gifs/gif/png/{name}/*.png"),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    data.append({"gif": gif, "png": pngs})

# Write the json
with open("static/gifs.json", "w") as f:
    json.dump(data, f)
