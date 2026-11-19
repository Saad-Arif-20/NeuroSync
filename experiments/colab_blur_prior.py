# NeuroSync FYP - blur-prior image targets (idea from Wu et al. 2025, UBP, arXiv 2503.04207), THINGS-EEG2
# AI-ASSISTED: written by Claude (Anthropic) for Saad Arif; declare in the AI Use Statements.
# Idea: train the EEG encoder to match CLIP features of BLURRED training images (blur radius r), because EEG carries
# little high-frequency detail. The paper gives no numeric sigma for radius r; here (my choice, documented) the blur is a
# Gaussian kernel of size r on the 224x224 preprocessed tensor with sigma = 0.3*((r-1)*0.5-1)+0.8 (OpenCV default rule).
# Everything else = current best pipeline (CNN encoder, rep-aug, 40 epochs, final epoch, 200-way retrieval).
# PHASE A (validation only; no test data): subjects 1-3, seeds 0-1, 20 held-out training concepts (200 images).
#   For each r in RADII: train on blurred-r targets; evaluate against UNBLURRED candidate features (the headline) and
#   against matched blurred-r candidates (secondary, for insight). Decision rule fixed in advance: r* = radius with the best
#   mean validation top-1 against UNBLURRED candidates; run phase B only if r* != 0 (i.e. beats no blur).
# PHASE B (test): subjects 1-10, 3 seeds, targets blurred with r*, unblurred test candidates; single-seed + 3-seed ensemble.
import os, glob, json, time, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
sh = lambda c: subprocess.run(c, shell=True, check=True)
URL = {1:"nb8wr",2:"u43p5",3:"fy4n6",4:"py8jz",5:"2cn7v",6:"bd83f",7:"mw8hr",8:"n9bzm",9:"fv6jz",10:"8fyu9"}
SIZE = {1:1071722467,2:1071516660,3:1071567672,4:1071421033,5:1072655693,6:1072667739,7:1071671310,8:1071848309,9:1071469818,10:1071595380}
BASE = [25.7, 23.8, 28.0, 27.8, 20.5, 36.5, 31.5, 39.2, 30.7, 41.5]      # E10 CNN baseline per-subject mean top-1
E12S = [29.0, 26.0, 29.7, 32.7, 24.2, 33.0, 32.3, 40.0, 31.2, 42.3]      # E12 rep-aug single-seed per-subject mean top-1
RADII = [0, 5, 11, 21, 31]
sigma_of = lambda r: 0.3 * ((r - 1) * 0.5 - 1) + 0.8
dev = "cuda"; os.makedirs("data", exist_ok=True)
for f, u in (("data/image_metadata.npy", "qkgtf"),):
    if not os.path.exists(f): sh(f"wget -q -O {f} https://osf.io/download/{u}/")
meta = np.load("data/image_metadata.npy", allow_pickle=True).item()

# ---- CLIP ViT-B/32 (LAION-2B) features of images blurred at each radius, one decode pass ----
if not os.path.exists("data/feats_blur.pt"):
    sh("pip -q install open_clip_torch")
    sh("wget -q -O data/test_images.zip https://osf.io/download/znu7b/; wget -q -O data/training_images.zip https://osf.io/download/3v527/")
    sh("unzip -q -o data/test_images.zip -d data/img; unzip -q -o data/training_images.zip -d data/img")
    import open_clip
    from PIL import Image
    paths = {f: os.path.join(r, f) for r, _, fs in os.walk("data/img") for f in fs if f.endswith(".jpg")}
    tf, ef = list(meta["train_img_files"]), list(meta["test_img_files"])
    print("images matched:", sum(f in paths for f in tf + ef), "of", len(tf) + len(ef))
    cm, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); cm = cm.cuda().eval().half()
    class DS(torch.utils.data.Dataset):
        def __init__(s, files): s.files = files
        def __len__(s): return len(s.files)
        def __getitem__(s, i): return pre(Image.open(paths[s.files[i]]).convert("RGB"))
    @torch.no_grad()
    def extract(files):
        out = {r: [] for r in RADII}
        for x in torch.utils.data.DataLoader(DS(files), batch_size=256, num_workers=2):
            x = x.cuda()
            for r in RADII:
                xb = x if r == 0 else TF.gaussian_blur(x, [r, r], [sigma_of(r)] * 2)
                out[r].append(F.normalize(cm.encode_image(xb.half()).float(), dim=-1).cpu())
        return {r: torch.cat(v) for r, v in out.items()}
    torch.save({"train": extract(tf), "test": extract(ef)}, "data/feats_blur.pt"); del cm; torch.cuda.empty_cache()
FB = torch.load("data/feats_blur.pt"); FTR, FTE = FB["train"], FB["test"]
print("blur features:", {r: tuple(FTR[r].shape) for r in RADII}, tuple(FTE[0].shape), "| start", time.strftime("%H:%M:%S"))
concepts = np.array(meta["train_img_concepts"]); vcon = np.random.RandomState(123).choice(np.unique(concepts), 20, replace=False)
isval = np.isin(concepts, vcon); vt = torch.tensor(isval, device=dev); vcpu = torch.tensor(isval)

class Enc(nn.Module):
    def __init__(s, ch=17, T=100, d=512):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(1, 40, (1, 25), padding=(0, 12)), nn.BatchNorm2d(40), nn.ELU(),
                              nn.Conv2d(40, 40, (ch, 1)), nn.BatchNorm2d(40), nn.ELU(), nn.AvgPool2d((1, 5)), nn.Dropout(0.5), nn.Flatten())
        s.proj = nn.Sequential(nn.Linear(40 * (T // 5), d), nn.GELU(), nn.Dropout(0.5), nn.Linear(d, d))
        s.logit_scale = nn.Parameter(torch.tensor(2.659))
    def forward(s, x): return F.normalize(s.proj(s.net(x.unsqueeze(1))), dim=-1)

def batch(X4, idx):   # rep-aug: average a random 1-4 of the 4 repetitions (E12)
    x = X4[idx]; k = torch.randint(1, 5, (len(idx),), device=dev)
    mask = (torch.rand(len(idx), 4, device=dev).argsort(1).argsort(1) < k[:, None]).float()
    return (x * mask[:, :, None, None]).sum(1) / k[:, None, None].float()

def train(X4, Y, seed, shuffle=False, epochs=40, bs=500):
    torch.manual_seed(seed); np.random.seed(seed); Y = Y.to(dev)
    if shuffle: Y = Y[torch.randperm(len(Y), device=dev)]
    m = Enc().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X4), device=dev)
        for i in range(0, len(X4), bs):
            idx = perm[i:i+bs]; z = m(batch(X4, idx)); y = Y[idx]
            lg = m.logit_scale.exp().clamp(max=100) * z @ y.T; lab = torch.arange(len(idx), device=dev)
            loss = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def emb(m, X): m.eval(); return m(X.to(dev)).cpu()
def acc(z, Y):
    rank = (z @ Y.T).argsort(1, descending=True); gt = torch.arange(len(Y)).unsqueeze(1)
    return (rank[:, :1] == gt).any(1).float().mean().item() * 100, (rank[:, :5] == gt).any(1).float().mean().item() * 100

def load(s):
    z = f"data/sub-{s:02d}.zip"; d = f"data/sub-{s:02d}"
    if not os.path.isdir(d):
        sh(f"wget -q -O {z} https://osf.io/download/{URL[s]}/"); assert os.path.getsize(z) == SIZE[s]
        sh(f"unzip -q -o {z} -d {d}; rm -f {z}")
    tr = np.load(glob.glob(f"{d}/**/preprocessed_eeg_training.npy", recursive=True)[0], allow_pickle=True).item()
    te = np.load(glob.glob(f"{d}/**/preprocessed_eeg_test.npy", recursive=True)[0], allow_pickle=True).item()
    return torch.tensor(tr["preprocessed_eeg_data"], dtype=torch.float32).to(dev), torch.tensor(te["preprocessed_eeg_data"].mean(1), dtype=torch.float32)

# ---------------- Phase A: validation only ----------------
vA = {r: [] for r in RADII}; vM = {r: [] for r in RADII}
for s in (1, 2, 3):
    X4, _ = load(s); Xv = X4[vt].mean(1); Xt = X4[~vt]
    for seed in (0, 1):
        for r in RADII:
            z = emb(train(Xt, FTR[r][~vcpu], seed), Xv)
            a0 = acc(z, FTR[0][vcpu]); aM = acc(z, FTR[r][vcpu]); vA[r].append(a0[0]); vM[r].append(aM[0])
            print(f"VAL S{s:02d} seed{seed} r={r:2d}: unblurred-gallery {a0[0]:.1f}/{a0[1]:.1f} | matched-gallery {aM[0]:.1f}/{aM[1]:.1f}", flush=True)
    del X4, Xv, Xt; torch.cuda.empty_cache()
print("PHASE A mean val top-1 by radius (unblurred gallery | matched gallery):")
for r in RADII: print(f"  r={r:2d} (sigma {sigma_of(r) if r else 0:.1f}): {np.mean(vA[r]):.1f} | {np.mean(vM[r]):.1f}")
rstar = max(RADII, key=lambda r: np.mean(vA[r])); gain = np.mean(vA[rstar]) - np.mean(vA[0])
print(f"PHASE A: r*={rstar}, gain over r=0 {gain:+.1f} -> {'RUN phase B' if rstar != 0 else 'STOP (no blur beats none)'}")

# ---------------- Phase B: test, only if a blur radius beat no blur on validation ----------------
if rstar != 0:
    res = []
    for s in range(1, 11):
        X4, Xte = load(s); zs = []
        for seed in (0, 1, 2):
            z = emb(train(X4, FTR[rstar], seed), Xte); zs.append(z); a, b = acc(z, FTE[0]); print(f"S{s:02d} blur r={rstar} seed{seed}: {a:.1f}/{b:.1f}", flush=True)
        ea, eb = acc(F.normalize(sum(zs), dim=-1), FTE[0]); sa = np.mean([acc(z, FTE[0])[0] for z in zs]); sb = np.mean([acc(z, FTE[0])[1] for z in zs])
        res.append((s, sa, sb, ea, eb)); print(f"S{s:02d} single-mean {sa:.1f}/{sb:.1f} | ensemble {ea:.1f}/{eb:.1f}  [{time.strftime('%H:%M:%S')}]", flush=True)
        if s == 1:
            ca, cb = acc(emb(train(X4, FTR[rstar], 0, shuffle=True), Xte), FTE[0]); print(f"S01 shuffled-label control: {ca:.1f}/{cb:.1f}", flush=True)
        json.dump(res, open("data/results_blur.json", "w")); del X4; torch.cuda.empty_cache()
    R = np.array(res); B = np.array(BASE); E = np.array(E12S)
    print("SUMMARY subject | single top1/top5 | ensemble top1/top5 | E12 single | E10 base")
    for r, e, b in zip(R, E, B): print(f"S{int(r[0]):02d}  {r[1]:5.1f} / {r[2]:5.1f} | {r[3]:5.1f} / {r[4]:5.1f} | {e:5.1f} | {b:5.1f}")
    print(f"ACROSS (r*={rstar}): single top1 {R[:,1].mean():.1f}+/-{R[:,1].std(ddof=1):.1f} top5 {R[:,2].mean():.1f}+/-{R[:,2].std(ddof=1):.1f} | ensemble top1 {R[:,3].mean():.1f}+/-{R[:,3].std(ddof=1):.1f} top5 {R[:,4].mean():.1f}+/-{R[:,4].std(ddof=1):.1f}")
    d1, d2 = R[:, 1] - E, R[:, 1] - B
    print(f"PAIRED single top1 vs E12 rep-aug {d1.mean():+.1f}+/-{d1.std(ddof=1):.1f} (better {(d1>0).sum()}/10) | vs E10 baseline {d2.mean():+.1f}+/-{d2.std(ddof=1):.1f} (better {(d2>0).sum()}/10)")
