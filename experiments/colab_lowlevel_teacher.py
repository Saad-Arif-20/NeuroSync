# NeuroSync FYP - E15: low-level / appearance teacher (DINOv2 intermediate layers, 16x16 pixel thumbnail) fused with CLIP, THINGS-EEG2
# AI-ASSISTED: written by Claude (Anthropic) for Saad Arif; declare in the AI Use Statements.
# Why: published ablations (Learning Brain Representation with Hierarchical Visual Embeddings, arXiv 2602.07495; Complementarity-Driven
# Dual-Teacher Hierarchical Alignment, arXiv 2611.12798 - single-source preprint) report that adding low-level visual targets to semantic CLIP
# targets gives the largest gains. EEG from occipital/parietal channels may carry low-level information that CLIP-only targets underuse.
# Same EEG pipeline as E12 (CNN, rep-aug, InfoNCE, 40 epochs, final epoch). One encoder per teacher; fusion at test time on similarity matrices
# (each EEG row z-scored, then w * S_other + (1-w) * S_clip).
# PHASE A (validation only, NO test data): subjects 1-3, seeds 0-1, 20 held-out concepts (200 images). Teachers: clip, DINOv2-base CLS at layers
#   2,4,6,8,10,12, pix16. Candidates = (teacher, w in 0.25/0.5/0.75). Decision rule fixed in advance: run phase B only if the best fused
#   validation top-1 beats clip-only by >= 2.0 points (about the noise level of this small validation set).
# PHASE B (test): subjects 1-10, 3 seeds per teacher, 3-seed embedding ensemble per teacher, fused with the chosen (teacher, w); S01 shuffled control.
import os, glob, json, time, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sh = lambda c: subprocess.run(c, shell=True, check=True)
URL = {1:"nb8wr",2:"u43p5",3:"fy4n6",4:"py8jz",5:"2cn7v",6:"bd83f",7:"mw8hr",8:"n9bzm",9:"fv6jz",10:"8fyu9"}
SIZE = {1:1071722467,2:1071516660,3:1071567672,4:1071421033,5:1072655693,6:1072667739,7:1071671310,8:1071848309,9:1071469818,10:1071595380}
E12ENS = [30.0, 26.5, 33.0, 33.0, 25.5, 34.5, 33.5, 42.5, 31.5, 46.0]   # E12 3-seed ensemble per-subject top-1 (CLIP ViT-B/32 only)
LAYERS = [2, 4, 6, 8, 10, 12]; WGRID = [0.25, 0.5, 0.75]; THRESH = 2.0
dev = "cuda"; os.makedirs("data", exist_ok=True)
print("start", time.strftime("%Y-%m-%d %H:%M:%S"), torch.cuda.get_device_name(0), flush=True)
if not os.path.exists("data/image_metadata.npy"): sh("wget -q -O data/image_metadata.npy https://osf.io/download/qkgtf/")
meta = np.load("data/image_metadata.npy", allow_pickle=True).item()
tf, ef = list(meta["train_img_files"]), list(meta["test_img_files"])
need_clip = not os.path.exists("data/f_ViT-B-32_train.npy")
need_dino = not os.path.exists("data/f_dino_train.npz")
need_pix = not os.path.exists("data/f_pix_train.npy")
if need_clip or need_dino or need_pix:
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
    if need_pix:
        tfp = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.Resize(16), transforms.ToTensor()])
        for tag, files in (("test", ef), ("train", tf)):
            out = [x.flatten(1) for x in loader(files, tfp)]
            np.save("data/f_pix_%s.npy" % tag, torch.cat(out).numpy())
        print("pixel features done", flush=True)

def prep(tr, te):   # centre by the TRAIN mean, then L2-normalise (same constants for train and test candidates)
    mu = tr.mean(0, keepdims=True)
    return F.normalize(torch.tensor(tr - mu), dim=-1), F.normalize(torch.tensor(te - mu), dim=-1)
TG = {"clip": (torch.tensor(np.load("data/f_ViT-B-32_train.npy")), torch.tensor(np.load("data/f_ViT-B-32_test.npy")))}
dtr, dte = np.load("data/f_dino_train.npz"), np.load("data/f_dino_test.npz")
for l in LAYERS: TG["d%d" % l] = prep(dtr["l%d" % l], dte["l%d" % l])
TG["pix"] = prep(np.load("data/f_pix_train.npy"), np.load("data/f_pix_test.npy"))
KEYS = list(TG.keys()); print("teachers:", KEYS, [tuple(TG[k][0].shape) for k in KEYS], flush=True)
concepts = np.array(meta["train_img_concepts"]); vcon = np.random.RandomState(123).choice(np.unique(concepts), 20, replace=False)
isval = np.isin(concepts, vcon); print("val images:", int(isval.sum()), "train images:", int((~isval).sum()), flush=True)

class Enc(nn.Module):
    def __init__(s, d, ch=17, T=100):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12)), nn.BatchNorm2d(40), nn.ELU(),
            nn.Conv2d(40, 40, (ch, 1)), nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d((1, 5)), nn.Dropout(0.5), nn.Flatten())
        s.proj = nn.Sequential(nn.Linear(40 * (T // 5), 512), nn.GELU(), nn.Dropout(0.5), nn.Linear(512, d))
        s.logit_scale = nn.Parameter(torch.tensor(2.659))
    def forward(s, x): return F.normalize(s.proj(s.net(x.unsqueeze(1))), dim=-1)

def batch(X4, idx):
    x = X4[idx]; k = torch.randint(1, 5, (len(idx),), device=dev)
    mask = (torch.rand(len(idx), 4, device=dev).argsort(1).argsort(1) < k[:, None]).float()
    return (x * mask[:, :, None, None]).sum(1) / k[:, None, None].float()

def train(X4, Y, seed, shuffle=False, epochs=40, bs=500):
    torch.manual_seed(seed); np.random.seed(seed); Y = Y.to(dev)
    if shuffle: Y = Y[torch.randperm(len(Y), device=dev)]
    m = Enc(Y.shape[1]).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
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
def accS(S):
    rank = S.argsort(1, descending=True); gt = torch.arange(S.shape[0]).unsqueeze(1)
    return (rank[:, :1] == gt).any(1).float().mean().item() * 100, (rank[:, :5] == gt).any(1).float().mean().item() * 100
def zrow(S): return (S - S.mean(1, keepdim=True)) / S.std(1, keepdim=True)
def fuse(Sa, Sb, w): return (1 - w) * zrow(Sa) + w * zrow(Sb)

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
        for k in KEYS:
            Ytr = TG[k][0]; z = emb(train(Xt, Ytr[~vi], seed), Xv); SV[(s, seed, k)] = z @ Ytr[vi].T
            a, b = accS(SV[(s, seed, k)]); print("VAL S%02d seed%d %-5s %.1f/%.1f" % (s, seed, k, a, b), flush=True)
    del X4, Xv, Xt; torch.cuda.empty_cache()
runs = [(s, sd) for s in (1, 2, 3) for sd in (0, 1)]
single = {k: np.mean([accS(SV[(s, sd, k)])[0] for s, sd in runs]) for k in KEYS}
print("PHASE A single-teacher mean val top-1:")
for k in KEYS: print("  %-5s %.1f" % (k, single[k]))
fz = {}
for k in KEYS:
    if k == "clip": continue
    for w in WGRID: fz[(k, w)] = np.mean([accS(fuse(SV[(s, sd, "clip")], SV[(s, sd, k)], w))[0] for s, sd in runs])
print("PHASE A fused with clip (mean val top-1), columns w=0.25 / 0.5 / 0.75:")
for k in KEYS:
    if k != "clip": print("  clip+%-5s %.1f / %.1f / %.1f" % (k, fz[(k, 0.25)], fz[(k, 0.5)], fz[(k, 0.75)]))
(bk, bw) = max(fz, key=fz.get); gain = fz[(bk, bw)] - single["clip"]
print("PHASE A: best = clip + %s at w=%.2f, fused %.1f vs clip-only %.1f, gain %+.1f (threshold +%.1f) -> %s" % (bk, bw, fz[(bk, bw)], single["clip"], gain, THRESH, "RUN phase B" if gain >= THRESH else "STOP (no teacher passes)"), flush=True)

# ---------------- Phase B: test, only if phase A passes the pre-set threshold ----------------
if gain >= THRESH:
    res = []
    for s in range(1, 11):
        X4, Xte = load(s); E = {}
        for k in ("clip", bk):
            zs = [emb(train(X4, TG[k][0], seed), Xte) for seed in (0, 1, 2)]
            E[k] = F.normalize(sum(zs), dim=-1) @ TG[k][1].T
        ca, cb = accS(E["clip"]); da, db = accS(E[bk]); fa, fb = accS(fuse(E["clip"], E[bk], bw))
        res.append((s, ca, cb, da, db, fa, fb)); print("S%02d clip-ens %.1f/%.1f | %s-ens %.1f/%.1f | fused %.1f/%.1f [%s]" % (s, ca, cb, bk, da, db, fa, fb, time.strftime("%H:%M:%S")), flush=True)
        if s == 1:
            C = {k: emb(train(X4, TG[k][0], 0, shuffle=True), Xte) @ TG[k][1].T for k in ("clip", bk)}
            xa, xb = accS(fuse(C["clip"], C[bk], bw)); print("S01 shuffled-label control (fused): %.1f/%.1f" % (xa, xb), flush=True)
        json.dump(res, open("data/results_lowlevel.json", "w")); del X4; torch.cuda.empty_cache()
    R = np.array(res); B = np.array(E12ENS)
    print("SUMMARY clip+%s w=%.2f | subject | clip-ens | %s-ens | fused (top1/top5) | E12 clip-ens top1" % (bk, bw, bk))
    for r, b in zip(R, B): print("S%02d  %5.1f/%5.1f | %5.1f/%5.1f | %5.1f/%5.1f | %5.1f" % (int(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], b))
    d = R[:, 5] - R[:, 1]
    print("ACROSS: clip-ens %.1f/%.1f | %s-ens %.1f/%.1f | fused %.1f+/-%.1f / %.1f+/-%.1f" % (R[:,1].mean(), R[:,2].mean(), bk, R[:,3].mean(), R[:,4].mean(), R[:,5].mean(), R[:,5].std(ddof=1), R[:,6].mean(), R[:,6].std(ddof=1)))
    print("PAIRED fused vs clip-ens top1: %+.1f+/-%.1f (better in %d/10)" % (d.mean(), d.std(ddof=1), (d > 0).sum()))
