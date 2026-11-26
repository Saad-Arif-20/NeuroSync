# NeuroSync FYP - E16: more / larger mid-level teachers and multi-teacher fusion on top of E15 (CLIP + DINOv2-base layer 6 = 53.0/84.8 test top-1/top-5)
# AI-ASSISTED: written by Claude (Anthropic) for Saad Arif; declare in the AI Use Statements.
# Why: E15 showed mid-level DINOv2 features as retrieval targets beat CLIP-only by a wide margin. Next question: do other mid-level layers, larger
# or smaller DINOv2 models, or a 3-way fusion add more? Same EEG pipeline as E12/E15. PHASE A (validation only, NO test data): subjects 1-3,
# seeds 0-1, 200 held-out images. Teachers: clip, d4, d6, d8 (DINOv2-base), cat (concat of normalised d4,d6,d8, one encoder), L8, L12 (DINOv2-large
# hidden_states 8,12), S6 (DINOv2-small hidden_states 6). Candidate fusions are fixed below. Decision rule fixed in advance: run phase B only if the
# best candidate beats the E15 configuration (clip+d6, w=0.75, re-computed here on the same runs) by >= 2.0 val top-1 points.
# PHASE B (test): subjects 1-10, 3 seeds per teacher, per-teacher embedding ensemble, best fusion; S01 shuffled control; paired vs E15 per-subject fused top-1.
import os, glob, json, time, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sh = lambda c: subprocess.run(c, shell=True, check=True)
URL = {1:"nb8wr",2:"u43p5",3:"fy4n6",4:"py8jz",5:"2cn7v",6:"bd83f",7:"mw8hr",8:"n9bzm",9:"fv6jz",10:"8fyu9"}
SIZE = {1:1071722467,2:1071516660,3:1071567672,4:1071421033,5:1072655693,6:1072667739,7:1071671310,8:1071848309,9:1071469818,10:1071595380}
E15F = [55.0, 47.5, 49.0, 50.0, 46.0, 58.0, 51.0, 59.5, 53.5, 60.5]   # E15 fused (clip+d6, w=0.75) per-subject test top-1
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
        for name, tagname, ls in (("facebook/dinov2-large", "dinoL", [8, 12]), ("facebook/dinov2-small", "dinoS", [6])):
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
Ltr, Lte = np.load("data/f_dinoL_train.npz"), np.load("data/f_dinoL_test.npz"); Str, Ste = np.load("data/f_dinoS_train.npz"), np.load("data/f_dinoS_test.npz")
TG["L8"] = prep(Ltr["l8"], Lte["l8"]); TG["L12"] = prep(Ltr["l12"], Lte["l12"]); TG["S6"] = prep(Str["l6"], Ste["l6"])
TG["cat"] = (F.normalize(torch.cat([TG["d4"][0], TG["d6"][0], TG["d8"][0]], 1), dim=-1), F.normalize(torch.cat([TG["d4"][1], TG["d6"][1], TG["d8"][1]], 1), dim=-1))
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
def fusew(Ss, W): return sum(w * zrow(Ss[k]) for k, w in W.items())
CANDS = {"E15 clip.25+d6.75": {"clip": .25, "d6": .75}}
for X in ("d4", "d8", "L8", "L12", "S6"): CANDS["clip.2+d6.5+%s.3" % X] = {"clip": .2, "d6": .5, X: .3}
CANDS["clip.25+d4.25+d6.25+d8.25"] = {"clip": .25, "d4": .25, "d6": .25, "d8": .25}
for X in ("cat", "L8", "L12", "S6", "d4", "d8"): CANDS["clip.25+%s.75" % X] = {"clip": .25, X: .75}
CANDS["clip.25+L8.375+d6.375"] = {"clip": .25, "L8": .375, "d6": .375}

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
            a_, b_ = accS(SV[(s, seed, k)]); print("VAL S%02d seed%d %-5s %.1f/%.1f" % (s, seed, k, a_, b_), flush=True)
    del X4, Xv, Xt; torch.cuda.empty_cache()
runs = [(s, sd) for s in (1, 2, 3) for sd in (0, 1)]
print("PHASE A single-teacher mean val top-1:")
for k in KEYS: print("  %-5s %.1f" % (k, np.mean([accS(SV[(s, sd, k)])[0] for s, sd in runs])))
fz = {}
for name, W in CANDS.items(): fz[name] = np.mean([accS(fusew({k: SV[(s, sd, k)] for k in W}, W))[0] for s, sd in runs])
print("PHASE A fusion candidates (mean val top-1):")
for name in CANDS: print("  %-28s %.1f" % (name, fz[name]))
base = fz["E15 clip.25+d6.75"]; bn = max(fz, key=fz.get); gain = fz[bn] - base; bW = CANDS[bn]
print("PHASE A: best = %s, val %.1f vs E15 config %.1f, gain %+.1f (threshold +%.1f) -> %s" % (bn, fz[bn], base, gain, THRESH, "RUN phase B" if gain >= THRESH else "STOP (E15 config stays)"), flush=True)

# ---------------- Phase B: test, only if phase A passes the pre-set threshold ----------------
if gain >= THRESH:
    res = []
    for s in range(1, 11):
        X4, Xte = load(s); E = {}
        for k in bW:
            zs = [emb(train(X4, TG[k][0], seed), Xte) for seed in (0, 1, 2)]
            E[k] = F.normalize(sum(zs), dim=-1) @ TG[k][1].T
        fa, fb = accS(fusew(E, bW)); res.append((s, fa, fb)); print("S%02d fused %.1f/%.1f | E15 %.1f [%s]" % (s, fa, fb, E15F[s - 1], time.strftime("%H:%M:%S")), flush=True)
        if s == 1:
            C = {k: emb(train(X4, TG[k][0], 0, shuffle=True), Xte) @ TG[k][1].T for k in bW}
            xa, xb = accS(fusew(C, bW)); print("S01 shuffled-label control (fused): %.1f/%.1f" % (xa, xb), flush=True)
        json.dump(res, open("data/results_e16.json", "w")); del X4; torch.cuda.empty_cache()
    R = np.array(res); B = np.array(E15F); d = R[:, 1] - B
    print("ACROSS: %s fused %.1f+/-%.1f / %.1f+/-%.1f" % (bn, R[:,1].mean(), R[:,1].std(ddof=1), R[:,2].mean(), R[:,2].std(ddof=1)))
    print("PAIRED vs E15 top1: %+.1f+/-%.1f (better in %d/10)" % (d.mean(), d.std(ddof=1), (d > 0).sum()))
