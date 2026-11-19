# NeuroSync FYP - multi-subject EEG->image retrieval (THINGS-EEG2), subjects 1-10
# Run in Google Colab (T4 GPU). One cell. Same encoder/loss/protocol as colab_baseline_full.py (subject 1).
# AI-ASSISTED: written by Claude (Anthropic) for Saad Arif; declare in the AI Use Statements.
# Protocol: per subject, average repetitions (4 train / 80 test), 40 epochs, final epoch only,
# 200-way retrieval vs CLIP ViT-B/32 (LAION-2B) features, 3 seeds + 1 shuffled-label control (seed 0).
import os, glob, json, time, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sh = lambda c: subprocess.run(c, shell=True, check=True)
os.makedirs("data", exist_ok=True)
URL = {1:"nb8wr",2:"u43p5",3:"fy4n6",4:"py8jz",5:"2cn7v",6:"bd83f",7:"mw8hr",8:"n9bzm",9:"fv6jz",10:"8fyu9"}
SIZE = {1:1071722467,2:1071516660,3:1071567672,4:1071421033,5:1072655693,6:1072667739,7:1071671310,8:1071848309,9:1071469818,10:1071595380}
dev = "cuda"
print("start", time.strftime("%Y-%m-%d %H:%M:%S"), torch.cuda.get_device_name(0))

# ---- CLIP features (built once, in metadata order = EEG index order) ----
if not (os.path.exists("data/train_feat.npy") and os.path.exists("data/test_feat.npy")):
    sh("pip -q install open_clip_torch")
    sh("wget -q -O data/image_metadata.npy https://osf.io/download/qkgtf/")
    sh("wget -q -O data/test_images.zip https://osf.io/download/znu7b/")
    sh("wget -q -O data/training_images.zip https://osf.io/download/3v527/")
    sh("unzip -q -o data/test_images.zip -d data/img; unzip -q -o data/training_images.zip -d data/img")
    import open_clip
    from PIL import Image
    meta = np.load("data/image_metadata.npy", allow_pickle=True).item()
    paths = {f: os.path.join(r, f) for r, _, fs in os.walk("data/img") for f in fs if f.endswith(".jpg")}
    tf, ef = list(meta["train_img_files"]), list(meta["test_img_files"])
    print("images matched:", sum(f in paths for f in tf + ef), "of", len(tf) + len(ef))
    cm, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    cm = cm.cuda().eval().half()
    @torch.no_grad()
    def embed(files, bs=256):
        out = []
        for i in range(0, len(files), bs):
            x = torch.stack([pre(Image.open(paths[f]).convert("RGB")) for f in files[i:i+bs]]).cuda().half()
            out.append(F.normalize(cm.encode_image(x).float(), dim=-1).cpu())
        return torch.cat(out).numpy()
    np.save("data/test_feat.npy", embed(ef)); np.save("data/train_feat.npy", embed(tf))
    del cm; torch.cuda.empty_cache()
Ytr = torch.tensor(np.load("data/train_feat.npy")); Yte = torch.tensor(np.load("data/test_feat.npy"))
print("features", tuple(Ytr.shape), tuple(Yte.shape))

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

def evaluate(m, Xte):
    m.eval()
    with torch.no_grad():
        z = m(Xte.to(dev)).cpu(); rank = (z @ Yte.T).argsort(1, descending=True); gt = torch.arange(200).unsqueeze(1)
        return (rank[:, :1] == gt).any(1).float().mean().item() * 100, (rank[:, :5] == gt).any(1).float().mean().item() * 100

def run(Xtr, Xte, seed, shuffle, epochs=40, bs=500):
    torch.manual_seed(seed); np.random.seed(seed)
    X, Y = Xtr.to(dev), Ytr.to(dev)
    if shuffle: Y = Y[torch.randperm(len(Y), device=dev)]
    m = Enc().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), bs):
            idx = perm[i:i+bs]; z = m(X[idx]); y = Y[idx]
            lg = m.logit_scale.exp().clamp(max=100) * z @ y.T; lab = torch.arange(len(idx), device=dev)
            loss = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(m, Xte)

results = []   # (subject, kind, seed, top1, top5)
for s in range(1, 11):
    z = f"data/sub-{s:02d}.zip"; d = f"data/sub-{s:02d}"
    if not os.path.isdir(d):
        sh(f"wget -q -O {z} https://osf.io/download/{URL[s]}/")
        assert os.path.getsize(z) == SIZE[s], f"size mismatch sub {s}"
        sh(f"unzip -q -o {z} -d {d}; rm -f {z}")
    tr = np.load(glob.glob(f"{d}/**/preprocessed_eeg_training.npy", recursive=True)[0], allow_pickle=True).item()
    te = np.load(glob.glob(f"{d}/**/preprocessed_eeg_test.npy", recursive=True)[0], allow_pickle=True).item()
    assert tr["preprocessed_eeg_data"].shape == (16540, 4, 17, 100) and te["preprocessed_eeg_data"].shape == (200, 80, 17, 100)
    Xtr = torch.tensor(tr["preprocessed_eeg_data"].mean(1), dtype=torch.float32)
    Xte = torch.tensor(te["preprocessed_eeg_data"].mean(1), dtype=torch.float32)
    for seed in (0, 1, 2):
        a, b = run(Xtr, Xte, seed, False); results.append((s, "real", seed, a, b))
        print(f"S{s:02d} real seed{seed}: {a:.1f}/{b:.1f}", flush=True)
    a, b = run(Xtr, Xte, 0, True); results.append((s, "shuffled", 0, a, b))
    print(f"S{s:02d} shuffled: {a:.1f}/{b:.1f}  [{time.strftime('%H:%M:%S')}]", flush=True)
    json.dump(results, open("data/results_multisubject.json", "w"))
    del tr, te, Xtr, Xte

print("\nSUBJECT SUMMARY (mean over 3 seeds; top1/top5 %), shuffled control")
m1 = []
for s in range(1, 11):
    r = [x for x in results if x[0] == s and x[1] == "real"]; c = [x for x in results if x[0] == s and x[1] == "shuffled"][0]
    t1, t5 = np.mean([x[3] for x in r]), np.mean([x[4] for x in r]); m1.append((t1, t5))
    print(f"S{s:02d}  {t1:5.1f} / {t5:5.1f}   (min-max top1 {min(x[3] for x in r):.1f}-{max(x[3] for x in r):.1f})   ctrl {c[3]:.1f}/{c[4]:.1f}")
m1 = np.array(m1)
print(f"ACROSS SUBJECTS: top1 {m1[:,0].mean():.1f} +/- {m1[:,0].std(ddof=1):.1f} | top5 {m1[:,1].mean():.1f} +/- {m1[:,1].std(ddof=1):.1f}  (chance 0.5/2.5)")
