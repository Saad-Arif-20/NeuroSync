# NeuroSync FYP - E19: spatial patch-token targets with a conv adapter (RPA-style) and low-level descriptors, on top of E18 (plain 80.0/97.1)
# AI-ASSISTED: written by Claude (Anthropic) for Saad Arif; declare in the AI Use Statements.
# Why (literature, preprints): RPA (arXiv 2609.31698) reports CLS-token target 69.5 -> mean of patch tokens 77.0 -> conv adapter over the patch grid 95.4
# (same layer, same encoder); AttrEcho (2611.09123) reports CLIP 76.0 -> 94.8 when HOG/texture/contour descriptors are added. Both are single-source claims
# and RPA selected its layer on test accuracy. We test the ideas on OUR data with validation-only selection.
# New targets: P_L8, P_L12 = DINOv2-large layer 8/12 patch tokens (16x16 grid avg-pooled 2x2 to 8x8, 1024-d), P_d6 = DINOv2-base layer 6 patch tokens (8x8, 768-d),
# PM_L12 = control: mean of P_L12 tokens + linear (no spatial adapter). attr = HOG + LBP + Canny descriptors (243-d); clipA = concat(clip, attr)/sqrt2.
# Adapter: 1x1 conv D->256, 3x3 conv 256->256, 1x1 conv 256->D (each GroupNorm(32)+GELU), residual add, global average pool, linear to 1024.
# Brain encoder, recipe and CLS teachers are exactly E18 mlp0 (25 epochs). Fusion sets are fixed below. Selection on PLAIN validation only.
# PHASE A (validation only, NO test data): subjects 1-3, seeds 0-1, 200 held-out images. Decision rule fixed in advance: run phase B only if the best fusion
# beats the E18 configuration (clip .2, cat .4, L12 .4; recomputed here on the same runs) by >= 2.0 val top-1 points.
# PHASE B (test): subjects 1-10, 3 seeds per teacher, similarity matrices averaged; plain and CSLS(k=1) reported separately; S01 shuffled-label control;
# paired comparison against E18 per-subject plain and CSLS.
import os, glob, json, time, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sh = lambda c: subprocess.run(c, shell=True, check=True)
URL = {1:"nb8wr",2:"u43p5",3:"fy4n6",4:"py8jz",5:"2cn7v",6:"bd83f",7:"mw8hr",8:"n9bzm",9:"fv6jz",10:"8fyu9"}
SIZE = {1:1071722467,2:1071516660,3:1071567672,4:1071421033,5:1072655693,6:1072667739,7:1071671310,8:1071848309,9:1071469818,10:1071595380}
E18P = [82.0, 77.0, 76.5, 77.5, 74.0, 84.0, 77.0, 86.0, 81.0, 85.0]   # E18 plain per-subject test top-1
E18C = [95.0, 90.5, 92.0, 88.0, 88.0, 95.5, 90.5, 94.0, 94.5, 96.5]   # E18 CSLS(k=1) per-subject test top-1
THRESH = 2.0; dev = "cuda"; os.makedirs("data", exist_ok=True)
print("start", time.strftime("%Y-%m-%d %H:%M:%S"), torch.cuda.get_device_name(0), flush=True)
if not os.path.exists("data/image_metadata.npy"): sh("wget -q -O data/image_metadata.npy https://osf.io/download/qkgtf/")
meta = np.load("data/image_metadata.npy", allow_pickle=True).item()
tf, ef = list(meta["train_img_files"]), list(meta["test_img_files"])
need_img = not os.path.isdir("data/img")
if need_img:
    sh("wget -q -O data/test_images.zip https://osf.io/download/znu7b/"); sh("wget -q -O data/training_images.zip https://osf.io/download/3v527/")
    sh("unzip -q -o data/test_images.zip -d data/img; unzip -q -o data/training_images.zip -d data/img")
paths = {f: os.path.join(r, f) for r, _, fs in os.walk("data/img") for f in fs if f.endswith(".jpg")}
print("images matched:", sum(f in paths for f in tf + ef), "of", len(tf) + len(ef), flush=True)
from PIL import Image
from torchvision import transforms
class DS(torch.utils.data.Dataset):
    def __init__(s, files, pre): s.f, s.p = files, pre
    def __len__(s): return len(s.f)
    def __getitem__(s, i): return s.p(Image.open(paths[s.f[i]]).convert("RGB"))
def loader(files, pre): return torch.utils.data.DataLoader(DS(files, pre), batch_size=128, num_workers=2)
tfm = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# ---- patch-token features (cached) ----
for name, tag, ls in (("facebook/dinov2-large", "PL", [8, 12]), ("facebook/dinov2-base", "Pb", [6])):
    if os.path.exists("data/p_%s_l%d_test.npy" % (tag, ls[0])): continue
    from transformers import AutoModel
    dm = AutoModel.from_pretrained(name).cuda().eval().half()
    for split, files in (("test", ef), ("train", tf)):
        outs = {l: [] for l in ls}
        with torch.no_grad():
            for x in loader(files, tfm):
                hs = dm(pixel_values=x.cuda().half(), output_hidden_states=True).hidden_states
                for l in ls:
                    p = hs[l][:, 1:, :]; B, N, D = p.shape; g = int(N ** 0.5)
                    p = p.reshape(B, g, g, D).permute(0, 3, 1, 2); p = F.avg_pool2d(p.float(), 2).permute(0, 2, 3, 1)
                    outs[l].append(p.half().cpu())
        for l in ls: np.save("data/p_%s_l%d_%s.npy" % (tag, l, split), torch.cat(outs[l]).numpy())
        del outs
    del dm; torch.cuda.empty_cache(); print(name, "patch features done", flush=True)
# ---- CLS features from E16 caches (clip, dino base 4/6/8, dino large 8/12) ----
if not os.path.exists("data/f_ViT-B-32_train.npy") or not os.path.exists("data/f_dino_train.npz") or not os.path.exists("data/f_dinoL_train.npz"):
    raise SystemExit("CLS feature caches missing (f_ViT-B-32_*, f_dino_*, f_dinoL_*): run colab_multiteacher.py (E16) first")
# ---- low-level descriptors (cached) ----
if not os.path.exists("data/attr_train.npy"):
    from skimage.feature import hog, local_binary_pattern, canny
    from joblib import Parallel, delayed
    def desc(f):
        g = np.asarray(Image.open(paths[f]).convert("L").resize((128, 128)), dtype=np.float32) / 255.0
        h = hog(g, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(1, 1), feature_vector=True)
        lbp = local_binary_pattern((g * 255).astype(np.uint8), 8, 1, "uniform"); e = canny(g, sigma=2.0).astype(np.float32); v = []
        for i in range(3):
            for j in range(3):
                sl = (slice(i * 42, (i + 1) * 42 if i < 2 else 128), slice(j * 42, (j + 1) * 42 if j < 2 else 128))
                v.extend(np.bincount(lbp[sl].astype(int).ravel(), minlength=10)[:10] / lbp[sl].size); v.append(e[sl].mean())
        return np.concatenate([h, np.array(v, dtype=np.float32)]).astype(np.float32)
    for split, files in (("test", ef), ("train", tf)): np.save("data/attr_%s.npy" % split, np.stack(Parallel(n_jobs=4)(delayed(desc)(f) for f in files)))
    print("attr features done", flush=True)

def prep(tr, te):   # centre by the TRAIN mean, then L2-normalise
    mu = tr.mean(0, keepdims=True); return F.normalize(torch.tensor(tr - mu), dim=-1), F.normalize(torch.tensor(te - mu), dim=-1)
def prep_patch(tag, l):   # standardise per channel with TRAIN statistics; chunked to GPU fp16 (RAM-safe)
    tr = np.load("data/p_%s_l%d_train.npy" % (tag, l), mmap_mode="r"); te = np.load("data/p_%s_l%d_test.npy" % (tag, l), mmap_mode="r")
    sub = np.asarray(tr[::8]).astype(np.float32).reshape(-1, tr.shape[-1]); mu = torch.tensor(sub.mean(0)).to(dev); sd = torch.tensor(sub.std(0) + 1e-6).to(dev); del sub
    def f(a):
        out = torch.empty(a.shape, dtype=torch.float16, device=dev)
        for i in range(0, len(a), 1000): out[i:i+1000] = ((torch.tensor(np.asarray(a[i:i+1000])).to(dev).float() - mu) / sd).half()
        return out
    return f(tr), f(te)
TG = {"clip": prep(np.load("data/f_ViT-B-32_train.npy"), np.load("data/f_ViT-B-32_test.npy"))}
dtr, dte = np.load("data/f_dino_train.npz"), np.load("data/f_dino_test.npz")
d4, d6, d8 = [prep(dtr["l%d" % l], dte["l%d" % l]) for l in (4, 6, 8)]
TG["cat"] = (F.normalize(torch.cat([d4[0], d6[0], d8[0]], 1), dim=-1), F.normalize(torch.cat([d4[1], d6[1], d8[1]], 1), dim=-1))
Ltr, Lte = np.load("data/f_dinoL_train.npz"), np.load("data/f_dinoL_test.npz"); TG["L12"] = prep(Ltr["l12"], Lte["l12"])
at = prep(np.load("data/attr_train.npy"), np.load("data/attr_test.npy"))
TG["clipA"] = (F.normalize(torch.cat([TG["clip"][0], at[0]], 1), dim=-1), F.normalize(torch.cat([TG["clip"][1], at[1]], 1), dim=-1))
PT = {}
PT["P_L8"] = prep_patch("PL", 8); PT["P_L12"] = prep_patch("PL", 12); PT["P_d6"] = prep_patch("Pb", 6)
for k in PT: TG[k] = PT[k]
TG["PM_L12"] = PT["P_L12"]
KIND = {k: "vec" for k in ("clip", "cat", "L12", "clipA")}; KIND.update({"P_L8": "patch", "P_L12": "patch", "P_d6": "patch", "PM_L12": "pmean"})
print("teachers:", {k: tuple(TG[k][0].shape) for k in TG}, flush=True)
concepts = np.array(meta["train_img_concepts"]); vcon = np.random.RandomState(123).choice(np.unique(concepts), 20, replace=False)
isval = np.isin(concepts, vcon); print("val images:", int(isval.sum()), "train images:", int((~isval).sum()), flush=True)

class Net(nn.Module):   # E18 mlp0 brain encoder + a target head (linear for vectors, RPA-style adapter for patch grids, mean+linear for the control)
    def __init__(s, shape, kind, ch=17, T=100, h=1024):
        super().__init__(); s.kind = kind
        s.inp = nn.Linear(ch * T, h); s.mlp = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h)); s.ln = nn.LayerNorm(h)
        s.logit_scale = nn.Parameter(torch.tensor(2.659))
        if kind == "vec": s.tproj = nn.Linear(shape[-1], h)
        else:
            D = shape[-1]; s.tproj = nn.Linear(D, h)
            if kind == "patch":
                s.c1 = nn.Conv2d(D, 256, 1); s.g1 = nn.GroupNorm(32, 256); s.c2 = nn.Conv2d(256, 256, 3, padding=1); s.g2 = nn.GroupNorm(32, 256)
                s.c3 = nn.Conv2d(256, D, 1); s.g3 = nn.GroupNorm(32, D); s.act = nn.GELU()
    def brain(s, x): h = s.inp(x.flatten(1)); return F.normalize(s.ln(h + s.mlp(h)), dim=-1)
    def target(s, y):
        if s.kind == "vec": return F.normalize(s.tproj(y), dim=-1)
        x = y.float().permute(0, 3, 1, 2)
        if s.kind == "patch":
            u = s.act(s.g1(s.c1(x))); u = s.act(s.g2(s.c2(u))); u = s.act(s.g3(s.c3(u))); x = x + u
        return F.normalize(s.tproj(x.mean((2, 3))), dim=-1)

def batch(X4, idx):
    x = X4[idx]; k = torch.randint(1, 5, (len(idx),), device=dev)
    mask = (torch.rand(len(idx), 4, device=dev).argsort(1).argsort(1) < k[:, None]).float()
    return (x * mask[:, :, None, None]).sum(1) / k[:, None, None].float()

def train(X4, Y, kind, seed, shuffle=False, epochs=25, bs=1024):
    torch.manual_seed(seed); np.random.seed(seed); Y = Y.to(dev)
    if shuffle: Y = Y[torch.randperm(len(Y), device=dev)]
    m = Net(tuple(Y.shape[1:]), kind).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-4); steps = epochs * ((len(X4) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / 10 if t < 10 else 0.5 * (1 + np.cos(np.pi * (t - 10) / max(1, steps - 10))))
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X4), device=dev)
        for i in range(0, len(X4), bs):
            idx = perm[i:i+bs]; z = m.brain(batch(X4, idx)); y = m.target(Y[idx])
            lg = m.logit_scale.exp().clamp(max=100) * z @ y.T; lab = torch.arange(len(idx), device=dev)
            loss = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m

@torch.no_grad()
def sim(m, X, Ygal): m.eval(); return (m.brain(X.to(dev)) @ m.target(Ygal.to(dev)).T).cpu()
def accS(S):
    rank = S.argsort(1, descending=True); gt = torch.arange(S.shape[0]).unsqueeze(1)
    return (rank[:, :1] == gt).any(1).float().mean().item() * 100, (rank[:, :5] == gt).any(1).float().mean().item() * 100
def zrow(S): return (S - S.mean(1, keepdim=True)) / S.std(1, keepdim=True)
def fusew(Ss, W): return sum(w * zrow(Ss[k]) for k, w in W.items())
def csls(S, k): return 2 * S - S.topk(k, 1).values.mean(1, keepdim=True) - S.topk(k, 0).values.mean(0, keepdim=True)
FUS = {"E18w": {"clip": .2, "cat": .4, "L12": .4},
       "A": {"clip": .15, "cat": .3, "L12": .25, "P_L12": .3}, "B": {"clip": .15, "cat": .3, "L12": .25, "P_L8": .3},
       "C": {"clip": .15, "cat": .25, "L12": .2, "P_L8": .2, "P_L12": .2}, "D": {"clipA": .2, "cat": .4, "L12": .4},
       "E": {"P_L8": .5, "P_L12": .5}, "F": {"clip": .1, "P_L8": .45, "P_L12": .45},
       "G": {"clipA": .15, "cat": .3, "L12": .2, "P_L12": .35}, "H": {"clip": .1, "P_L8": .3, "P_L12": .3, "P_d6": .3}}
TEACH = list(TG.keys())

def load(s):
    z = "data/sub-%02d.zip" % s; d = "data/sub-%02d" % s
    if not os.path.isdir(d):
        sh("wget -q -O %s https://osf.io/download/%s/" % (z, URL[s])); assert os.path.getsize(z) == SIZE[s]
        sh("unzip -q -o %s -d %s; rm -f %s" % (z, d, z))
    tr = np.load(glob.glob(d + "/**/preprocessed_eeg_training.npy", recursive=True)[0], allow_pickle=True).item()
    te = np.load(glob.glob(d + "/**/preprocessed_eeg_test.npy", recursive=True)[0], allow_pickle=True).item()
    return torch.tensor(tr["preprocessed_eeg_data"], dtype=torch.float32).to(dev), torch.tensor(te["preprocessed_eeg_data"].mean(1), dtype=torch.float32)

# ---------------- Phase A: validation only ----------------
vt = torch.tensor(isval, device=dev); vi = torch.tensor(isval)
SV = {}
for s in (1, 2, 3):
    X4, _ = load(s); Xv = X4[vt].mean(1); Xt = X4[~vt]
    for seed in (0, 1):
        for k in TEACH:
            Ytr = TG[k][0]; m = train(Xt, Ytr[~vt.to(Ytr.device)], KIND[k], seed); SV[(s, seed, k)] = sim(m, Xv, Ytr[vt.to(Ytr.device)])
            a_, b_ = accS(SV[(s, seed, k)]); print("VAL S%02d seed%d %-6s %.1f/%.1f" % (s, seed, k, a_, b_), flush=True); del m
    del X4, Xv, Xt; torch.cuda.empty_cache()
runs = [(s, sd) for s in (1, 2, 3) for sd in (0, 1)]
print("PHASE A single-teacher mean val top-1:")
for k in TEACH: print("  %-6s %.1f" % (k, np.mean([accS(SV[(s, sd, k)])[0] for s, sd in runs])))
fz = {}; fc = {}
for fn, W in FUS.items():
    Fr = {r: fusew({k: SV[(r[0], r[1], k)] for k in W}, W) for r in runs}
    fz[fn] = np.mean([accS(Fr[r])[0] for r in runs]); fc[fn] = np.mean([accS(csls(Fr[r], 1))[0] for r in runs])
print("PHASE A fusion sets (mean val top-1): plain | CSLS k=1 (info only)")
for fn in FUS: print("  %-5s %.1f | %.1f   %s" % (fn, fz[fn], fc[fn], FUS[fn]))
base = fz["E18w"]; bf = max(fz, key=fz.get); gain = fz[bf] - base
print("PHASE A: best (plain) = %s, val %.1f vs E18 config %.1f, gain %+.1f (threshold +%.1f) -> %s" % (bf, fz[bf], base, gain, THRESH, "RUN phase B" if gain >= THRESH else "STOP (E18 config stays)"), flush=True)

# ---------------- Phase B: test, only if phase A passes the pre-set threshold ----------------
if gain >= THRESH:
    W = FUS[bf]; res = []
    for s in range(1, 11):
        X4, Xte = load(s); E = {}
        for k in W: E[k] = sum(zrow(sim(train(X4, TG[k][0], KIND[k], seed), Xte, TG[k][1])) for seed in (0, 1, 2)) / 3
        Fz = fusew(E, W); pa, pb = accS(Fz); ca, cb = accS(csls(Fz, 1))
        res.append((s, pa, pb, ca, cb)); print("S%02d plain %.1f/%.1f | CSLS(k=1) %.1f/%.1f | E18 plain %.1f CSLS %.1f [%s]" % (s, pa, pb, ca, cb, E18P[s - 1], E18C[s - 1], time.strftime("%H:%M:%S")), flush=True)
        if s == 1:
            C = {k: sim(train(X4, TG[k][0], KIND[k], 0, shuffle=True), Xte, TG[k][1]) for k in W}
            xa, xb = accS(fusew(C, W)); print("S01 shuffled-label control (plain): %.1f/%.1f" % (xa, xb), flush=True)
        json.dump(res, open("data/results_e19.json", "w")); del X4; torch.cuda.empty_cache()
    R = np.array(res); d = R[:, 1] - np.array(E18P); d2 = R[:, 3] - np.array(E18C)
    print("ACROSS fusion %s: plain %.1f+/-%.1f / %.1f+/-%.1f | CSLS(k=1) %.1f+/-%.1f / %.1f+/-%.1f" % (bf, R[:,1].mean(), R[:,1].std(ddof=1), R[:,2].mean(), R[:,2].std(ddof=1), R[:,3].mean(), R[:,3].std(ddof=1), R[:,4].mean(), R[:,4].std(ddof=1)))
    print("PAIRED plain vs E18 plain: %+.1f+/-%.1f (better in %d/10) | CSLS vs E18 CSLS: %+.1f+/-%.1f (better in %d/10)" % (d.mean(), d.std(ddof=1), (d > 0).sum(), d2.mean(), d2.std(ddof=1), (d2 > 0).sum()))
