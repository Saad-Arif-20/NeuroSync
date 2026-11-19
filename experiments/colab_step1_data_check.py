# NeuroSync - Step 1: get THINGS-EEG2 subject 1 and check its shape.
# Run in Google Colab (Runtime > Change runtime type > T4 GPU), one "# %%" block per cell.
# Status: NOT yet run. Written 2 Nov 2026 from the OSF file listing; confirm the printed shapes below.
# Dataset: Gifford et al. 2022, THINGS-EEG2, OSF project 3jk45, licence file on the OSF image-set page.

# %% 1. Download subject 1 preprocessed EEG (~1.07 GB) and the image set
import os, zipfile, requests

os.makedirs("data", exist_ok=True)
FILES = {
    "sub-01.zip": "https://osf.io/download/nb8wr/",          # preprocessed EEG, subject 1
    "test_images.zip": "https://osf.io/download/znu7b/",     # 200 test images (~8 MB)
    "training_images.zip": "https://osf.io/download/3v527/", # 16,540 training images (~655 MB)
}
for name, url in FILES.items():
    path = f"data/{name}"
    if not os.path.exists(path):
        print("downloading", name)
        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    with zipfile.ZipFile(path) as z:
        z.extractall(f"data/{name[:-4]}")

# %% 2. Look at what the zips contain (file names are NOT assumed)
for root, _, files in os.walk("data"):
    for fn in files:
        if fn.endswith((".npy", ".npz")):
            p = os.path.join(root, fn)
            print(p, round(os.path.getsize(p) / 1e6, 1), "MB")

# %% 3. Load the EEG arrays and print shapes (paste this output back to Claude)
import numpy as np, glob

for p in sorted(glob.glob("data/sub-01/**/preprocessed_eeg_*.npy", recursive=True)):
    d = np.load(p, allow_pickle=True).item()
    print(p, "keys:", list(d.keys()))
    x = d["preprocessed_eeg_data"]
    print("  EEG shape:", x.shape, x.dtype)
    print("  channels:", d.get("ch_names"))
    t = d.get("times")
    print("  times:", None if t is None else (t[0], t[-1], len(t)))
# Expected (from the paper, to be CONFIRMED by this output):
#   training (16540 images, 4 repeats, 17 channels, 100 samples)
#   test     (200 images, 80 repeats, 17 channels, 100 samples)
