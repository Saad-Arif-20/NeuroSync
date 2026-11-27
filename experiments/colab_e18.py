# NeuroSync FYP - E18: spatial-filter-bank encoder, longer training and teacher sets with the MLP encoder (E17 plain = 74.9/95.1)
# AI-ASSISTED: written by Claude (Anthropic) for Saad Arif; declare in the AI Use Statements.
# Why: DTHA (arXiv 2611.12798, single-source preprint under review, 92.4 plain top-1) ablates its encoder: removing a K=16 positivity-constrained
# spatial filter bank costs about 22 points, and it trains 250 epochs with cosine; the NVOL/Shallow-Alignment papers use several mid layers. We test,
# on OUR data: (a) mlp0 (E17 winner) vs sbk (spatial bank K=16 with softplus weights over the 17 channels, depthwise temporal conv k=7, then the same
# MLP head), (b) 25 vs 100 epochs, (c) four fusion sets over teachers clip, d6 (DINOv2-base L6), L8 and L12 (DINOv2-large), cat (d4+d6+d8 concat).
# PHASE A (validation only, NO test data): subjects 1-3, seeds 0-1, 200 held-out images. Selection uses the PLAIN validation score only (CSLS k=1 is printed
# for information; it is transductive and is not used to choose anything). Decision rule fixed in advance: run phase B (test) only if the best
# (encoder, epochs, fusion) beats the E17 configuration (mlp0, 25 epochs, E16 weights) plain validation score by >= 2.0 points.
# PHASE B (test): subjects 1-10, 3 seeds per teacher, similarity matrices averaged; plain and CSLS(k=1) reported; S01 shuffled-label control;
# paired comparison against E17 per-subject plain (and CSLS) top-1.
import os, glob, json, time, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sh = lambda c: subprocess.run(c, shell=True, check=True)
URL = {1:"nb8wr",2:"u43p5",3:"fy4n6",4:"py8jz",5:"2cn7v",6:"bd83f",7:"mw8hr",8:"n9bzm",9:"fv6jz",10:"8fyu9"}
SIZE = {1:1071722467,2:1071516660,3:1071567672,4:1071421033,5:1072655693,6:1072667739,7:1071671310,8:1071848309,9:1071469818,10:1071595380}
E17P = [76.5, 70.5, 71.0, 71.5, 64.5, 76.0, 74.0, 83.5, 79.0, 83.0]   # E17 plain per-subject test top-1 (mlp0, E16 weights)
E17C = [89.0, 87.5, 87.0, 84.0, 81.0, 91.0, 84.5, 90.0, 88.0, 92.0]   # E17 with CSLS k=1 per-subject test top-1
LAYERS = [4, 6, 8]; THRESH = 2.0
dev = "cuda"; os.makedirs("data", exist_ok=True)
print("start", time.strftime("%Y-%m-%d %H:%M:%S"), torch.cuda.get_device_name(0), flush=True)
if not os.path.exists("data/image_metadata.npy"): sh("wget -q -O data/image_metadata.npy https://osf.io/download/qkgtf/")
meta = np.load("data/image_metadata.npy", allow_pickle=True).item()
tf, ef = list(meta["train_img_files"]), list(meta["test_img_files"])
need_clip = not os.path.exists("data/f_ViT-B-32_train.npy")
need_dino = not os.path.exists("data/f_dino_train.npz")
need_big = not os.path.exists("data/f_dinoL_train.npz")
if need_clip or need_dino or need_big:
    sh("pip -q install open_clip_torch")
    if not os.path.isdir("data/img"):
        sh("wget -q -O data/test_images.zip https://osf.io/download/znu7b/")
        sh("wget -q -O data/training_images.zip https://osf.io/download/3v527/")
        sh("unzip -q -o data/test_images.zip -d data/img; unzip -q -o data/training_images.zip -d data/img")
    from PIL import Image
    from torchvision import transforms
    paths = {f: os.path.join(r, f) for r, _, fs in os.walk("data/img") for f in fs if f.endswith(".jpg")}
    print("images matched:", sum(f in paths for f in tf + ef), "of", len(tf) + len(ef), flush=True)
    class DS(torch.utils.data.Dataset):
        def __init__(s, files, pre): s.f, s.p = files, pre
        def __len__(s): return len(s.f)
        def __getitem__(s, i): return s.p(Image.open(paths[s.f[i]]).convert("RGB"))
    def loader(files, pre): return torch.utils.data.DataLoader(DS(files, pre), batch_size=128, num_workers=2)
    if need_clip:
        import open_clip
        cm, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); cm = cm.cuda().eval().half()
        for tag, files in (("test", ef), ("train", tf)):
            out = []
            with torch.no_grad():
                for x in loader(files, pre): out.append(F.normalize(cm.encode_image(x.cuda().half()).float(), dim=-1).cpu())
            np.save("data/f_ViT-B-32_%s.npy" % tag, torch.cat(out).numpy())
        del cm; torch.cuda.empty_cache(); print("clip features done", flush=True)
    if need_dino:
        from transformers import AutoModel
        dm = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval().half()
        tfm = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        for tag, files in (("test", ef), ("train", tf)):
            outs = {l: [] for l in LAYERS}
            with torch.no_grad():
                for x in loader(files, tfm):
                    hs = dm(pixel_values=x.cuda().half(), output_hidden_states=True).hidden_states
                    for l in LAYERS: outs[l].append(hs[l][:, 0].float().cpu())
            np.savez("data/f_dino_%s.npz" % tag, **{"l%d" % l: torch.cat(outs[l]).numpy() for l in LAYERS})
        del dm; torch.cuda.empty_cache(); print("dino features done", flush=True)
    if need_big:
        from transformers import AutoModel
        tfm2 = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        for name, tagname, ls in (("facebook/dinov2-large", "dinoL", [8, 12]),):
            dm = AutoModel.from_pretrained(name).cuda().eval().half()
            for tag, files in (("test", ef), ("train", tf)):
                outs = {l: [] for l in ls}
                with torch.no_grad():
                    for x in loader(files, tfm2):
                        hs = dm(pixel_values=x.cuda().half(), output_hidden_states=True).hidden_states
                        for l in ls: outs[l].append(hs[l][:, 0].float().cpu())
                np.savez("data/f_%s_%s.npz" % (tagname, tag), **{"l%d" % l: torch.cat(outs[l]).numpy() for l in ls})
            del dm; torch.cuda.empty_cache(); print(name, "features done", flush=True)

def prep(tr, te):   # centre by the TRAIN mean, then L2-normalise (same constants for train and test candidates)
    mu = tr.mean(0, keepdims=True)
    return F.normalize(torch.tensor(tr - mu), dim=-1), F.normalize(torch.tensor(te - mu), dim=-1)
TG = {"clip": (torch.tensor(np.load("data/f_ViT-B-32_train.npy")), torch.tensor(np.load("data/f_ViT-B-32_test.npy")))}
dtr, dte = np.load("data/f_dino_train.npz"), np.load("data/f_dino_test.npz")
for l in LAYERS: TG["d%d" % l] = prep(dtr["l%d" % l], dte["l%d" % l])
Ltr, Lte = np.load("data/f_dinoL_train.npz"), np.load("data/f_dinoL_test.npz")
TG["L8"] = prep(Ltr["l8"], Lte["l8"]); TG["L12"] = prep(Ltr["l12"], Lte["l12"])
TG["cat"] = (F.normalize(torch.cat([TG["d4"][0], TG["d6"][0], TG["d8"][0]], 1), dim=-1), F.normalize(torch.cat([TG["d4"][1], TG["d6"][1], TG["d8"][1]], 1), dim=-1))
for _k in ("d4", "d8"): del TG[_k]
KEYS = list(TG.keys()); print("teachers:", KEYS, [tuple(TG[k][0].shape) for k in KEYS], flush=True)
concepts = np.array(meta["train_img_concepts"]); vcon = np.random.RandomState(123).choice(np.unique(concepts), 20, replace=False)
isval = np.isin(concepts, vcon); print("val images:", int(isval.sum()), "train images:", int((~isval).sum()), flush=True)

class MBP(nn.Module):   # E17 encoder: flatten -> linear 1024 -> residual 2-layer MLP -> LayerNorm; linear target projector
    def __init__(s, d, ch=17, T=100, h=1024, bank=False, K=16):
        super().__init__()
        s.bank = bank
        if bank:
            s.sw = nn.Parameter(torch.zeros(K, ch)); s.tc = nn.Conv1d(K, K, 7, padding=3, groups=K); s.act = nn.GELU(); fin = K * T
        else: fin = ch * T
        s.inp = nn.Linear(fin, h); s.mlp = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h)); s.ln = nn.LayerNorm(h)
        s.tproj = nn.Linear(d, h); s.logit_scale = nn.Parameter(torch.tensor(2.659))
    def brain(s, x):
        if s.bank: x = s.act(s.tc(torch.einsum("kc,bct->bkt", F.softplus(s.sw), x)))
        h = s.inp(x.flatten(1)); return F.normalize(s.ln(h + s.mlp(h)), dim=-1)
    def target(s, y): return F.normalize(s.tproj(y), dim=-1)

def batch(X4, idx):
    x = X4[idx]; k = torch.randint(1, 5, (len(idx),), device=dev)
    mask = (torch.rand(len(idx), 4, device=dev).argsort(1).argsort(1) < k[:, None]).float()
    return (x * mask[:, :, None, None]).sum(1) / k[:, None, None].float()

def train(X4, Y, seed, kind, shuffle=False):
    torch.manual_seed(seed); np.random.seed(seed); Y = Y.to(dev)
    if shuffle: Y = Y[torch.randperm(len(Y), device=dev)]
    epochs = 100 if kind.endswith("e100") else 25; bs = 1024
    m = MBP(Y.shape[1], bank=kind.startswith("sbk")).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-4); steps = epochs * ((len(X4) + bs - 1) // bs)
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
FUS = {"E16w": {"clip": .25, "L8": .375, "d6": .375}, "4way": {"clip": .2, "L8": .3, "d6": .3, "L12": .2},
       "cat": {"clip": .25, "cat": .75}, "catL12": {"clip": .2, "cat": .4, "L12": .4}}
TEACH = ["clip", "d6", "L8", "L12", "cat"]; KINDS = ["mlp0", "mlp0e100", "sbk", "sbke100"]

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
        for kind in KINDS:
            for k in TEACH:
                Ytr = TG[k][0]; m = train(Xt, Ytr[~vi], seed, kind); SV[(s, seed, kind, k)] = sim(m, Xv, Ytr[vi])
                a_, b_ = accS(SV[(s, seed, kind, k)]); print("VAL S%02d seed%d %-8s %-4s %.1f/%.1f" % (s, seed, kind, k, a_, b_), flush=True)
    del X4, Xv, Xt; torch.cuda.empty_cache()
runs = [(s, sd) for s in (1, 2, 3) for sd in (0, 1)]
print("PHASE A single-teacher mean val top-1 (encoder x teacher):")
for kind in KINDS: print("  %-8s " % kind + "  ".join("%s %.1f" % (k, np.mean([accS(SV[(s, sd, kind, k)])[0] for s, sd in runs])) for k in TEACH))
fz = {}; fc = {}
for kind in KINDS:
    for fn, W in FUS.items():
        Fr = {r: fusew({k: SV[(r[0], r[1], kind, k)] for k in W}, W) for r in runs}
        fz[(kind, fn)] = np.mean([accS(Fr[r])[0] for r in runs]); fc[(kind, fn)] = np.mean([accS(csls(Fr[r], 1))[0] for r in runs])
print("PHASE A fused mean val top-1: plain | CSLS k=1 (info only); columns = fusion sets %s" % list(FUS))
for kind in KINDS: print("  %-8s " % kind + " ".join("%.1f|%.1f" % (fz[(kind, fn)], fc[(kind, fn)]) for fn in FUS))
base = fz[("mlp0", "E16w")]; (bk, bf) = max(fz, key=fz.get); gain = fz[(bk, bf)] - base
print("PHASE A: best (plain) = %s + %s, val %.1f vs E17 config %.1f, gain %+.1f (threshold +%.1f) -> %s" % (bk, bf, fz[(bk, bf)], base, gain, THRESH, "RUN phase B" if gain >= THRESH else "STOP (E17 config stays)"), flush=True)

# ---------------- Phase B: test, only if phase A passes the pre-set threshold ----------------
if gain >= THRESH:
    W = FUS[bf]; res = []
    for s in range(1, 11):
        X4, Xte = load(s); E = {}
        for k in W:
            E[k] = sum(zrow(sim(train(X4, TG[k][0], seed, bk), Xte, TG[k][1])) for seed in (0, 1, 2)) / 3
        Fz = fusew(E, W); pa, pb = accS(Fz); ca, cb = accS(csls(Fz, 1))
        res.append((s, pa, pb, ca, cb)); print("S%02d plain %.1f/%.1f | CSLS(k=1) %.1f/%.1f | E17 plain %.1f CSLS %.1f [%s]" % (s, pa, pb, ca, cb, E17P[s - 1], E17C[s - 1], time.strftime("%H:%M:%S")), flush=True)
        if s == 1:
            C = {k: sim(train(X4, TG[k][0], 0, bk, shuffle=True), Xte, TG[k][1]) for k in W}
            xa, xb = accS(fusew(C, W)); print("S01 shuffled-label control (plain): %.1f/%.1f" % (xa, xb), flush=True)
        json.dump(res, open("data/results_e18.json", "w")); del X4; torch.cuda.empty_cache()
    R = np.array(res); d = R[:, 1] - np.array(E17P); d2 = R[:, 3] - np.array(E17C)
    print("ACROSS %s + %s: plain %.1f+/-%.1f / %.1f+/-%.1f | CSLS(k=1) %.1f+/-%.1f / %.1f+/-%.1f" % (bk, bf, R[:,1].mean(), R[:,1].std(ddof=1), R[:,2].mean(), R[:,2].std(ddof=1), R[:,3].mean(), R[:,3].std(ddof=1), R[:,4].mean(), R[:,4].std(ddof=1)))
    print("PAIRED plain vs E17 plain: %+.1f+/-%.1f (better in %d/10) | CSLS vs E17 CSLS: %+.1f+/-%.1f (better in %d/10)" % (d.mean(), d.std(ddof=1), (d > 0).sum(), d2.mean(), d2.std(ddof=1), (d2 > 0).sum()))
