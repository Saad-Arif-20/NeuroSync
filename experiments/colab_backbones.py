# NeuroSync FYP - E14: image-backbone comparison for the retrieval targets, THINGS-EEG2
# AI-ASSISTED: written by Claude (Anthropic) for Saad Arif; declare in the AI Use Statements.
# Same EEG pipeline as E12 (CNN encoder, rep-aug, symmetric InfoNCE, 40 epochs, final epoch only); only the image model
# that supplies the target/candidate features changes. Targets AND the 200 test candidates both come from the same backbone.
# Phase A (validation only, NO test data): subjects 1-3, seeds 0-1, 20 held-out concepts (200 images), 200-way retrieval.
#   Decision rule fixed in advance: pick best-validation backbone b*; run Phase B only if b* != ViT-B/32 (mean val top-1 higher).
# Phase B (test): subjects 1-10, 3 seeds, single + 3-seed embedding ensemble, shuffled-label control on S01.
import os, glob, json, time, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sh = lambda c: subprocess.run(c, shell=True, check=True)
URL = {1:"nb8wr",2:"u43p5",3:"fy4n6",4:"py8jz",5:"2cn7v",6:"bd83f",7:"mw8hr",8:"n9bzm",9:"fv6jz",10:"8fyu9"}
SIZE = {1:1071722467,2:1071516660,3:1071567672,4:1071421033,5:1072655693,6:1072667739,7:1071671310,8:1071848309,9:1071469818,10:1071595380}
BACK = [("ViT-B-32", "laion2b_s34b_b79k"), ("RN50", "openai"), ("ViT-B-16", "laion2b_s34b_b88k"), ("ViT-L-14", "laion2b_s32b_b82k")]
E12S = [29.0, 26.0, 29.7, 32.7, 24.2, 33.0, 32.3, 40.0, 31.2, 42.3]   # E12 single-seed per-subject top-1 (ViT-B/32)
dev = "cuda"; os.makedirs("data", exist_ok=True)
print("start", time.strftime("%Y-%m-%d %H:%M:%S"), torch.cuda.get_device_name(0), flush=True)
if not os.path.exists("data/image_metadata.npy"): sh("wget -q -O data/image_metadata.npy https://osf.io/download/qkgtf/")
meta = np.load("data/image_metadata.npy", allow_pickle=True).item()
tf, ef = list(meta["train_img_files"]), list(meta["test_img_files"])
need = [n for n, _ in BACK if not os.path.exists("data/f_%s_train.npy" % n)]
if need:
    sh("pip -q install open_clip_torch")
    if not os.path.isdir("data/img"):
        sh("wget -q -O data/test_images.zip https://osf.io/download/znu7b/")
        sh("wget -q -O data/training_images.zip https://osf.io/download/3v527/")
        sh("unzip -q -o data/test_images.zip -d data/img; unzip -q -o data/training_images.zip -d data/img")
    import open_clip
    from PIL import Image
    paths = {f: os.path.join(r, f) for r, _, fs in os.walk("data/img") for f in fs if f.endswith(".jpg")}
    print("images matched:", sum(f in paths for f in tf + ef), "of", len(tf) + len(ef), flush=True)
    class DS(torch.utils.data.Dataset):
        def __init__(s, files, pre): s.f, s.p = files, pre
        def __len__(s): return len(s.f)
        def __getitem__(s, i): return s.p(Image.open(paths[s.f[i]]).convert("RGB"))
    for n, w in BACK:
        if n not in need: continue
        cm, _, pre = open_clip.create_model_and_transforms(n, pretrained=w); cm = cm.cuda().eval().half()
        for tag, files in (("test", ef), ("train", tf)):
            out = []
            dl = torch.utils.data.DataLoader(DS(files, pre), batch_size=128, num_workers=2)
            with torch.no_grad():
                for x in dl: out.append(F.normalize(cm.encode_image(x.cuda().half()).float(), dim=-1).cpu())
            a = torch.cat(out).numpy(); np.save("data/f_%s_%s.npy" % (n, tag), a); print("features", n, tag, a.shape, flush=True)
        del cm; torch.cuda.empty_cache()
FE = {n: (torch.tensor(np.load("data/f_%s_train.npy" % n)), torch.tensor(np.load("data/f_%s_test.npy" % n))) for n, _ in BACK}
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
def acc(z, Y):
    rank = (z @ Y.T).argsort(1, descending=True); gt = torch.arange(len(Y)).unsqueeze(1)
    return (rank[:, :1] == gt).any(1).float().mean().item() * 100, (rank[:, :5] == gt).any(1).float().mean().item() * 100

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
val = {n: [] for n, _ in BACK}
for s in (1, 2, 3):
    X4, _ = load(s); Xv = X4[vt].mean(1); Xt = X4[~vt]
    for seed in (0, 1):
        for n, _ in BACK:
            Ytr = FE[n][0]; a, b = acc(emb(train(Xt, Ytr[~vi], seed), Xv), Ytr[vi]); val[n].append(a)
            print("VAL S%02d seed%d %-9s %.1f/%.1f" % (s, seed, n, a, b), flush=True)
    del X4, Xv, Xt; torch.cuda.empty_cache()
print("PHASE A mean val top-1 by backbone:")
for n, _ in BACK: print("  %-9s dim %4d : %.1f" % (n, FE[n][0].shape[1], np.mean(val[n])))
best = max([n for n, _ in BACK], key=lambda n: np.mean(val[n])); gain = np.mean(val[best]) - np.mean(val["ViT-B-32"])
print("PHASE A: b*=%s, gain over ViT-B-32 %+.1f -> %s" % (best, gain, "RUN phase B" if best != "ViT-B-32" else "STOP (ViT-B-32 stays)"), flush=True)

# ---------------- Phase B: test, only if a different backbone wins validation ----------------
if best != "ViT-B-32":
    Ytr, Yte = FE[best]; res = []
    for s in range(1, 11):
        X4, Xte = load(s); zs = []
        for seed in (0, 1, 2):
            z = emb(train(X4, Ytr, seed), Xte); zs.append(z); a, b = acc(z, Yte)
            print("S%02d %s seed%d: %.1f/%.1f" % (s, best, seed, a, b), flush=True)
        ens = F.normalize(sum(zs), dim=-1); ea, eb = acc(ens, Yte)
        sa = np.mean([acc(z, Yte)[0] for z in zs]); sb = np.mean([acc(z, Yte)[1] for z in zs])
        res.append((s, sa, sb, ea, eb)); print("S%02d single-mean %.1f/%.1f | ensemble %.1f/%.1f [%s]" % (s, sa, sb, ea, eb, time.strftime("%H:%M:%S")), flush=True)
        if s == 1:
            ca, cb = acc(emb(train(X4, Ytr, 0, shuffle=True), Xte), Yte); print("S01 shuffled-label control: %.1f/%.1f" % (ca, cb), flush=True)
        json.dump(res, open("data/results_backbone.json", "w")); del X4; torch.cuda.empty_cache()
    R = np.array(res); B = np.array(E12S)
    print("SUMMARY %s  subject | single top1/top5 | ensemble top1/top5 | E12 ViT-B-32 single top1" % best)
    for r, b in zip(R, B): print("S%02d  %5.1f / %5.1f | %5.1f / %5.1f | %5.1f" % (int(r[0]), r[1], r[2], r[3], r[4], b))
    d1 = R[:, 1] - B
    print("ACROSS: single top1 %.1f+/-%.1f top5 %.1f+/-%.1f | ensemble top1 %.1f+/-%.1f top5 %.1f+/-%.1f | E12 single top1 %.1f" % (R[:,1].mean(), R[:,1].std(ddof=1), R[:,2].mean(), R[:,2].std(ddof=1), R[:,3].mean(), R[:,3].std(ddof=1), R[:,4].mean(), R[:,4].std(ddof=1), B.mean()))
    print("PAIRED single top1 vs E12 ViT-B-32: %+.1f+/-%.1f (better in %d/10)" % (d1.mean(), d1.std(ddof=1), (d1 > 0).sum()))
