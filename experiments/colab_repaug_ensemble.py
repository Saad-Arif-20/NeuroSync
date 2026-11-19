# NeuroSync FYP - repetition-averaging augmentation + seed ensemble, THINGS-EEG2
# AI-ASSISTED: written by Claude (Anthropic) for Saad Arif; declare in the AI Use Statements.
# Phase A (validation, NO test data used): subjects 1-3, 2 seeds. Compare baseline (average all 4 repetitions per image)
#   vs rep-aug (each step, average a random 1-4 of the 4 repetitions). Trained on all training concepts except 20 held-out
#   concepts (200 images); evaluated by 200-way retrieval on those held-out images. Decision rule fixed in advance:
#   run phase B only if mean validation top-1 of rep-aug > baseline.
# Phase B (test): rep-aug on subjects 1-10, 3 seeds, full training set; single-seed and 3-seed-embedding-ensemble test
#   retrieval (test EEG = mean of 80 repetitions, 200-way). One shuffled-label control (subject 1).
import os, glob, json, time, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sh = lambda c: subprocess.run(c, shell=True, check=True)
URL = {1:"nb8wr",2:"u43p5",3:"fy4n6",4:"py8jz",5:"2cn7v",6:"bd83f",7:"mw8hr",8:"n9bzm",9:"fv6jz",10:"8fyu9"}
SIZE = {1:1071722467,2:1071516660,3:1071567672,4:1071421033,5:1072655693,6:1072667739,7:1071671310,8:1071848309,9:1071469818,10:1071595380}
BASE = [25.7, 23.8, 28.0, 27.8, 20.5, 36.5, 31.5, 39.2, 30.7, 41.5]   # baseline per-subject mean top-1 (evidence E10)
dev = "cuda"; os.makedirs("data", exist_ok=True)
if not os.path.exists("data/image_metadata.npy"): sh("wget -q -O data/image_metadata.npy https://osf.io/download/qkgtf/")
meta = np.load("data/image_metadata.npy", allow_pickle=True).item()
if not (os.path.exists("data/train_feat.npy") and os.path.exists("data/test_feat.npy")):
    sh("pip -q install open_clip_torch")
    sh("wget -q -O data/test_images.zip https://osf.io/download/znu7b/")
    sh("wget -q -O data/training_images.zip https://osf.io/download/3v527/")
    sh("unzip -q -o data/test_images.zip -d data/img; unzip -q -o data/training_images.zip -d data/img")
    import open_clip
    from PIL import Image
    paths = {f: os.path.join(r, f) for r, _, fs in os.walk("data/img") for f in fs if f.endswith(".jpg")}
    tf, ef = list(meta["train_img_files"]), list(meta["test_img_files"])
    print("images matched:", sum(f in paths for f in tf + ef), "of", len(tf) + len(ef))
    cm, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    cm = cm.cuda().eval().half()
    @torch.no_grad()
    def embed_img(files, bs=256):
        out = []
        for i in range(0, len(files), bs):
            x = torch.stack([pre(Image.open(paths[f]).convert("RGB")) for f in files[i:i+bs]]).cuda().half()
            out.append(F.normalize(cm.encode_image(x).float(), dim=-1).cpu())
        return torch.cat(out).numpy()
    np.save("data/test_feat.npy", embed_img(ef)); np.save("data/train_feat.npy", embed_img(tf))
    del cm; torch.cuda.empty_cache()
Ytr = torch.tensor(np.load("data/train_feat.npy")); Yte = torch.tensor(np.load("data/test_feat.npy"))
concepts = np.array(meta["train_img_concepts"]); vcon = np.random.RandomState(123).choice(np.unique(concepts), 20, replace=False)
isval = np.isin(concepts, vcon); print("val images:", int(isval.sum()), "train images:", int((~isval).sum()), "| start", time.strftime("%H:%M:%S"))

class Enc(nn.Module):
    def __init__(s, ch=17, T=100, d=512):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12)), nn.BatchNorm2d(40), nn.ELU(),
            nn.Conv2d(40, 40, (ch, 1)), nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d((1, 5)), nn.Dropout(0.5), nn.Flatten())
        s.proj = nn.Sequential(nn.Linear(40 * (T // 5), d), nn.GELU(), nn.Dropout(0.5), nn.Linear(d, d))
        s.logit_scale = nn.Parameter(torch.tensor(2.659))
    def forward(s, x): return F.normalize(s.proj(s.net(x.unsqueeze(1))), dim=-1)

def batch(X4, idx, aug):
    x = X4[idx]
    if not aug: return x.mean(1)
    k = torch.randint(1, 5, (len(idx),), device=dev)
    mask = (torch.rand(len(idx), 4, device=dev).argsort(1).argsort(1) < k[:, None]).float()
    return (x * mask[:, :, None, None]).sum(1) / k[:, None, None].float()

def train(X4, Y, seed, aug, shuffle=False, epochs=40, bs=500):
    torch.manual_seed(seed); np.random.seed(seed); Y = Y.to(dev)
    if shuffle: Y = Y[torch.randperm(len(Y), device=dev)]
    m = Enc().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X4), device=dev)
        for i in range(0, len(X4), bs):
            idx = perm[i:i+bs]; z = m(batch(X4, idx, aug)); y = Y[idx]
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
vt = torch.tensor(isval, device=dev); Yv = Ytr[torch.tensor(isval)]; Ytrn = Ytr[torch.tensor(~isval)]
val = {False: [], True: []}
for s in (1, 2, 3):
    X4, _ = load(s); Xv = X4[vt].mean(1); Xt = X4[~vt]
    for seed in (0, 1):
        for aug in (False, True):
            a, b = acc(emb(train(Xt, Ytrn, seed, aug), Xv), Yv); val[aug].append(a)
            print(f"VAL S{s:02d} seed{seed} {'rep-aug' if aug else 'baseline'}: {a:.1f}/{b:.1f}", flush=True)
    del X4, Xv, Xt; torch.cuda.empty_cache()
gain = np.mean(val[True]) - np.mean(val[False])
print(f"PHASE A: mean val top-1 baseline {np.mean(val[False]):.1f} | rep-aug {np.mean(val[True]):.1f} | gain {gain:+.1f} -> {'RUN phase B' if gain > 0 else 'STOP (no gain)'}")

# ---------------- Phase B: test, only if phase A shows a gain ----------------
if gain > 0:
    res = []
    for s in range(1, 11):
        X4, Xte = load(s); zs = []
        for seed in (0, 1, 2):
            z = emb(train(X4, Ytr, seed, True), Xte); zs.append(z); a, b = acc(z, Yte)
            print(f"S{s:02d} rep-aug seed{seed}: {a:.1f}/{b:.1f}", flush=True)
        ens = F.normalize(sum(zs), dim=-1); ea, eb = acc(ens, Yte)
        sa = np.mean([acc(z, Yte)[0] for z in zs]); sb = np.mean([acc(z, Yte)[1] for z in zs])
        res.append((s, sa, sb, ea, eb)); print(f"S{s:02d} single-mean {sa:.1f}/{sb:.1f} | ensemble {ea:.1f}/{eb:.1f}  [{time.strftime('%H:%M:%S')}]", flush=True)
        if s == 1:
            ca, cb = acc(emb(train(X4, Ytr, 0, True, shuffle=True), Xte), Yte); print(f"S01 shuffled-label control: {ca:.1f}/{cb:.1f}", flush=True)
        json.dump(res, open("data/results_repaug.json", "w")); del X4; torch.cuda.empty_cache()
    R = np.array(res)
    print("SUMMARY  subject | single top1/top5 | ensemble top1/top5 | baseline top1 (E10)")
    for r, b in zip(R, BASE): print(f"S{int(r[0]):02d}  {r[1]:5.1f} / {r[2]:5.1f} | {r[3]:5.1f} / {r[4]:5.1f} | {b:5.1f}")
    B = np.array(BASE)
    print(f"ACROSS: single top1 {R[:,1].mean():.1f}+/-{R[:,1].std(ddof=1):.1f} top5 {R[:,2].mean():.1f}+/-{R[:,2].std(ddof=1):.1f} | ensemble top1 {R[:,3].mean():.1f}+/-{R[:,3].std(ddof=1):.1f} top5 {R[:,4].mean():.1f}+/-{R[:,4].std(ddof=1):.1f} | baseline top1 {B.mean():.1f}")
    d1, d2 = R[:, 1] - B, R[:, 3] - B
    print(f"PAIRED top1 vs baseline: single {d1.mean():+.1f}+/-{d1.std(ddof=1):.1f} (better in {(d1>0).sum()}/10) | ensemble {d2.mean():+.1f}+/-{d2.std(ddof=1):.1f} (better in {(d2>0).sum()}/10)")
