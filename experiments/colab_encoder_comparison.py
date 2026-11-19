# NeuroSync FYP - encoder comparison, THINGS-EEG2 subjects 1-10
# Original NeuroSync CNN+Transformer (eeg/model.py EEGEncoder) vs the CNN baseline from colab_multisubject.py.
# Same protocol: repetition-averaged EEG, 40 epochs, AdamW lr 1e-3 wd 1e-2, batch 500, symmetric InfoNCE with learnable
# temperature, final epoch only, 200-way retrieval, seeds 0,1,2. Self-contained: rebuilds CLIP features and downloads
# subject data if missing. Baseline per-subject means are hard-coded from evidence E10.
# AI-ASSISTED: written by Claude (Anthropic) for Saad Arif; declare in the AI Use Statements.
import os, glob, json, math, time, subprocess, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sh = lambda c: subprocess.run(c, shell=True, check=True)
URL = {1:"nb8wr",2:"u43p5",3:"fy4n6",4:"py8jz",5:"2cn7v",6:"bd83f",7:"mw8hr",8:"n9bzm",9:"fv6jz",10:"8fyu9"}
SIZE = {1:1071722467,2:1071516660,3:1071567672,4:1071421033,5:1072655693,6:1072667739,7:1071671310,8:1071848309,9:1071469818,10:1071595380}
dev = "cuda"
os.makedirs("data", exist_ok=True)
if not (os.path.exists("data/train_feat.npy") and os.path.exists("data/test_feat.npy")):   # CLIP ViT-B/32 (LAION-2B) features, as in colab_multisubject.py
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
print("start", time.strftime("%Y-%m-%d %H:%M:%S"), torch.cuda.get_device_name(0))

# ---- original NeuroSync EEGEncoder, copied unchanged from eeg/model.py ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term); pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(0)]

class EEGEncoder(nn.Module):
    def __init__(self, num_channels=64, projection_dim=512):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=8, stride=2, padding=3), nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(2, 2))
        self.pos_encoder = PositionalEncoding(d_model=256)
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.projection = nn.Linear(256, projection_dim)
    def forward(self, x):
        f = self.conv_blocks(x).permute(2, 0, 1)
        f = self.transformer_encoder(self.pos_encoder(f))
        return self.projection(f.mean(dim=0))

class TransformerWrapper(nn.Module):   # adds L2 norm + learnable temperature so it plugs into the same loop
    def __init__(self, ch=17):
        super().__init__()
        self.enc = EEGEncoder(num_channels=ch, projection_dim=512)
        self.logit_scale = nn.Parameter(torch.tensor(2.659))
    def forward(self, x): return F.normalize(self.enc(x), dim=-1)

def evaluate(m, Xte):
    m.eval()
    with torch.no_grad():
        z = m(Xte.to(dev)).cpu(); rank = (z @ Yte.T).argsort(1, descending=True); gt = torch.arange(200).unsqueeze(1)
        return (rank[:, :1] == gt).any(1).float().mean().item() * 100, (rank[:, :5] == gt).any(1).float().mean().item() * 100

def run(Xtr, Xte, seed, epochs=40, bs=500):
    torch.manual_seed(seed); np.random.seed(seed)
    X, Y = Xtr.to(dev), Ytr.to(dev)
    m = TransformerWrapper().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), bs):
            idx = perm[i:i+bs]; z = m(X[idx]); y = Y[idx]
            lg = m.logit_scale.exp().clamp(max=100) * z @ y.T; lab = torch.arange(len(idx), device=dev)
            loss = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(m, Xte)

res = []   # (subject, seed, top1, top5)
for s in range(1, 11):
    z = f"data/sub-{s:02d}.zip"; d = f"data/sub-{s:02d}"
    if not os.path.isdir(d):
        sh(f"wget -q -O {z} https://osf.io/download/{URL[s]}/"); assert os.path.getsize(z) == SIZE[s]
        sh(f"unzip -q -o {z} -d {d}; rm -f {z}")
    tr = np.load(glob.glob(f"{d}/**/preprocessed_eeg_training.npy", recursive=True)[0], allow_pickle=True).item()
    te = np.load(glob.glob(f"{d}/**/preprocessed_eeg_test.npy", recursive=True)[0], allow_pickle=True).item()
    Xtr = torch.tensor(tr["preprocessed_eeg_data"].mean(1), dtype=torch.float32)
    Xte = torch.tensor(te["preprocessed_eeg_data"].mean(1), dtype=torch.float32)
    for seed in (0, 1, 2):
        a, b = run(Xtr, Xte, seed); res.append((s, seed, a, b)); print(f"S{s:02d} transformer seed{seed}: {a:.1f}/{b:.1f}", flush=True)
    json.dump(res, open("data/results_transformer.json", "w")); del tr, te, Xtr, Xte

# CNN-baseline per-subject mean top-1 (3 seeds), from the earlier multi-subject run (evidence E10), same protocol and seeds
BASE = [25.7, 23.8, 28.0, 27.8, 20.5, 36.5, 31.5, 39.2, 30.7, 41.5]
print("SUBJECT SUMMARY transformer (mean of 3 seeds top1/top5) | baseline top1 (E10) | diff (T-B)")
T, B = [], []
for s in range(1, 11):
    r = [x for x in res if x[0] == s]; t1, t5 = np.mean([x[2] for x in r]), np.mean([x[3] for x in r]); T.append((t1, t5))
    b1 = BASE[s - 1]; B.append(b1)
    print(f"S{s:02d}  {t1:5.1f} / {t5:5.1f}  | {b1:5.1f} | {t1 - b1:+5.1f}")
T = np.array(T); B = np.array(B); dif = T[:, 0] - B   # per-subject paired difference
print(f"ACROSS SUBJECTS transformer: top1 {T[:,0].mean():.1f} +/- {T[:,0].std(ddof=1):.1f} | top5 {T[:,1].mean():.1f} +/- {T[:,1].std(ddof=1):.1f}")
print(f"baseline top1 {B.mean():.1f} | paired diff top1 {dif.mean():+.1f} +/- {dif.std(ddof=1):.1f} | subjects where transformer better: {(dif > 0).sum()}/10")
