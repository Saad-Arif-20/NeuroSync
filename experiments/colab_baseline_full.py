# NeuroSync FYP - EEG->image retrieval baseline (THINGS-EEG2, subject 1)
# Run in Google Colab (T4 GPU), Mon 2 Nov 2026. Cells separated by "# %%".
# AI-ASSISTED: written by Claude (Anthropic) for Saad Arif; to be declared in the AI Use Statements.
# Saad must be able to explain every block (viva).

# %% Cell 1 - download subject 1 preprocessed EEG (OSF, 1,071,722,467 bytes) and unzip
# !mkdir -p data
# !wget -q -O data/sub-01.zip https://osf.io/download/nb8wr/
# !unzip -q -o data/sub-01.zip -d data/sub-01
# !find data -name "*.npy" | head -20

# %% Cell 2 - Step 1 go/no-go: load and check shapes
import numpy as np, glob
tr = np.load(glob.glob("data/sub-01/**/preprocessed_eeg_training.npy", recursive=True)[0], allow_pickle=True).item()
te = np.load(glob.glob("data/sub-01/**/preprocessed_eeg_test.npy", recursive=True)[0], allow_pickle=True).item()
print(tr["preprocessed_eeg_data"].shape)   # observed (16540, 4, 17, 100)
print(te["preprocessed_eeg_data"].shape)   # observed (200, 80, 17, 100)
print(tr["ch_names"], tr["times"][[0, -1]]) # 17 channels; -0.2 to 0.79 s
print(np.isnan(tr["preprocessed_eeg_data"]).sum(), tr["preprocessed_eeg_data"].mean(), tr["preprocessed_eeg_data"].std())  # observed 0 NaN, mean -0.035, std ~0.84

# %% Cell 3 - Step 2 setup: open_clip, T4 check, images and metadata
# !pip -q install open_clip_torch
# !wget -q -O data/image_metadata.npy https://osf.io/download/qkgtf/
# !wget -q -O data/test_images.zip https://osf.io/download/znu7b/
# !wget -q -O data/training_images.zip https://osf.io/download/3v527/
# !unzip -q -o data/test_images.zip -d data/img
# !unzip -q -o data/training_images.zip -d data/img
# !nvidia-smi --query-gpu=name,memory.total --format=csv    # observed Tesla T4, 15360 MiB

# %% Cell 4 - metadata inspection
import numpy as np, torch, open_clip, os
meta = np.load("data/image_metadata.npy", allow_pickle=True).item()
for k, v in meta.items():
    print(k, type(v).__name__, len(v) if hasattr(v, "__len__") else v)
print(meta["train_img_files"][:3]); print(meta["test_img_files"][:3])

# %% Cell 5 - CLIP ViT-B/32 (LAION-2B) image features, in metadata order (= EEG index order)
from PIL import Image
paths = {f: os.path.join(r, f) for r, _, fs in os.walk("data/img") for f in fs if f.endswith(".jpg")}
train_files, test_files = list(meta["train_img_files"]), list(meta["test_img_files"])
print(len(paths), len(train_files), len(test_files), sum(f in paths for f in train_files + test_files))  # observed 16740 16540 200 16740
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.cuda().eval().half()

@torch.no_grad()
def embed(files, bs=256):
    out = []
    for i in range(0, len(files), bs):
        x = torch.stack([preprocess(Image.open(paths[f]).convert("RGB")) for f in files[i:i+bs]]).cuda().half()
        out.append(torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).cpu())
    return torch.cat(out).numpy()

test_feat = embed(test_files); train_feat = embed(train_files)
np.save("data/test_feat.npy", test_feat); np.save("data/train_feat.npy", train_feat)
print(test_feat.shape, train_feat.shape)   # observed (200, 512) (16540, 512)

# %% Cell 6 - contrastive EEG encoder + 200-way retrieval + shuffled-label control
import glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
ptr = glob.glob("data/sub-01/**/preprocessed_eeg_training.npy", recursive=True)[0]
pte = glob.glob("data/sub-01/**/preprocessed_eeg_test.npy", recursive=True)[0]
tr = np.load(ptr, allow_pickle=True).item(); te = np.load(pte, allow_pickle=True).item()
Xtr = torch.tensor(tr["preprocessed_eeg_data"].mean(1), dtype=torch.float32)   # (16540,17,100) avg of 4 reps
Xte = torch.tensor(te["preprocessed_eeg_data"].mean(1), dtype=torch.float32)   # (200,17,100) avg of 80 reps
Ytr = torch.tensor(np.load("data/train_feat.npy")); Yte = torch.tensor(np.load("data/test_feat.npy"))
print(Xtr.shape, Xte.shape, Ytr.shape, Yte.shape)
dev = "cuda"

class Enc(nn.Module):
    def __init__(s, ch=17, T=100, d=512):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12)), nn.BatchNorm2d(40), nn.ELU(),
            nn.Conv2d(40, 40, (ch, 1)), nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d((1, 5)), nn.Dropout(0.5), nn.Flatten())
        s.proj = nn.Sequential(nn.Linear(40 * (T // 5), d), nn.GELU(), nn.Dropout(0.5), nn.Linear(d, d))
        s.logit_scale = nn.Parameter(torch.tensor(2.659))
    def forward(s, x):
        return F.normalize(s.proj(s.net(x.unsqueeze(1))), dim=-1)

def evaluate(m):
    m.eval()
    with torch.no_grad():
        z = m(Xte.to(dev)).cpu(); sim = z @ Yte.T
        rank = sim.argsort(1, descending=True); gt = torch.arange(200).unsqueeze(1)
        top1 = (rank[:, :1] == gt).any(1).float().mean().item() * 100
        top5 = (rank[:, :5] == gt).any(1).float().mean().item() * 100
    return top1, top5

def run(shuffle, epochs=40, bs=500):
    X, Y = Xtr.to(dev), Ytr.to(dev)
    if shuffle: Y = Y[torch.randperm(len(Y), device=dev)]
    m = Enc().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), bs):
            idx = perm[i:i+bs]; z = m(X[idx]); y = Y[idx]
            lg = m.logit_scale.exp().clamp(max=100) * z @ y.T
            lab = torch.arange(len(idx), device=dev)
            loss = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0: print("shuffle" if shuffle else "real", ep + 1, round(loss.item(), 3), evaluate(m))
    return evaluate(m)

real = run(False); ctrl = run(True)
print("FINAL (top1%, top5%)  real:", real, " shuffled-control:", ctrl, " chance: (0.5, 2.5)")
# Observed: real top-1 30.0, top-5 ~61; shuffled control top-1 0.5, top-5 ~3
