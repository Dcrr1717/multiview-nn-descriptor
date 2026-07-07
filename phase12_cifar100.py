"""
╔══════════════════════════════════════════════════════════════════════╗
║  TESIS ESPOCH — FASE 12: VALIDACIÓN MULTIDOMINIO (CIFAR-100, n=40)   ║
║                                                                      ║
║  Entrena el MISMO zoo de arquitecturas (30 MLP + 10 CNN) sobre       ║
║  CIFAR-100 y extrae los descriptores X1 (grafo funcional) y X2       ║
║  (homología persistente). Resolución 32x32 idéntica → arquitecturas  ║
║  drop-in; cambia solo la dificultad de la tarea (10 -> 100 clases).  ║
║  Guarda los registros para el análisis leak-free (phase11).          ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import os, time, json, pickle, warnings
import numpy as np
import networkx as nx
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
try:
    from ripser import ripser as ripser_fn; HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

warnings.filterwarnings("ignore")
np.random.seed(2024); torch.manual_seed(2024)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("./data")
OUT = Path("./outputs/fase12_cifar100"); OUT.mkdir(parents=True, exist_ok=True)
PROG = OUT / "progress.json"
NC = 100  # CIFAR-100
T0 = time.time()
print("="*70); print(f"  FASE 12: CIFAR-100 (n=40)  | device={DEVICE}"); print("="*70)

# ── Datos ─────────────────────────────────────────────────────────────
MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
tfm_tr = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
tfm_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
tr_ds = datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=tfm_tr)
te_ds = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=tfm_te)
tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
te_ld = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
print(f"  CIFAR-100: {len(tr_ds)} train | {len(te_ds)} test")

# ══ ARQUITECTURAS MLP (de phase8) ════════════════════════════════════
class MLPShallow(nn.Module):
    def __init__(s, hidden_dims, dropout=0.2, input_dim=3072, n_classes=NC):
        super().__init__(); dims=[input_dim]+hidden_dims+[n_classes]; L=[]
        for i in range(len(dims)-2): L+=[nn.Linear(dims[i],dims[i+1]),nn.ReLU(),nn.Dropout(dropout)]
        L+=[nn.Linear(dims[-2],dims[-1])]; s.net=nn.Sequential(*L); s._acts={}; s._hooks=[]
    def _attach(s):
        for i,m in enumerate(s.net):
            if isinstance(m,nn.ReLU): s._hooks.append(m.register_forward_hook(
                lambda mo,inp,out,i=i: s._acts.update({f"relu_{i}":out.detach().cpu()})))
    def _detach(s):
        for h in s._hooks: h.remove()
        s._hooks=[]
    def forward(s,x): return s.net(x)

class MLPDeep(nn.Module):
    def __init__(s, hidden_dims, dropout=0.3, input_dim=3072, n_classes=NC):
        super().__init__(); dims=[input_dim]+hidden_dims+[n_classes]; L=[]
        for i in range(len(dims)-2): L+=[nn.Linear(dims[i],dims[i+1]),nn.BatchNorm1d(dims[i+1]),nn.GELU(),nn.Dropout(dropout)]
        L+=[nn.Linear(dims[-2],dims[-1])]; s.net=nn.Sequential(*L); s._acts={}; s._hooks=[]
    def _attach(s):
        for i,m in enumerate(s.net):
            if isinstance(m,nn.GELU): s._hooks.append(m.register_forward_hook(
                lambda mo,inp,out,i=i: s._acts.update({f"gelu_{i}":out.detach().cpu()})))
    def _detach(s):
        for h in s._hooks: h.remove()
        s._hooks=[]
    def forward(s,x): return s.net(x)

class ResBlock(nn.Module):
    def __init__(s, dim, dropout=0.2):
        super().__init__()
        s.block=nn.Sequential(nn.Linear(dim,dim),nn.BatchNorm1d(dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim,dim),nn.BatchNorm1d(dim))
        s.act=nn.GELU()
    def forward(s,x): return s.act(x+s.block(x))

class MLPResidual(nn.Module):
    def __init__(s, hidden_dim=512, n_blocks=4, dropout=0.2, input_dim=3072, n_classes=NC):
        super().__init__(); s.proj=nn.Linear(input_dim,hidden_dim)
        s.blocks=nn.ModuleList([ResBlock(hidden_dim,dropout) for _ in range(n_blocks)])
        s.head=nn.Linear(hidden_dim,n_classes); s._acts={}; s._hooks=[]
    def _attach(s):
        for i,blk in enumerate(s.blocks): s._hooks.append(blk.register_forward_hook(
            lambda mo,inp,out,i=i: s._acts.update({f"resblock_{i}":out.detach().cpu()})))
    def _detach(s):
        for h in s._hooks: h.remove()
        s._hooks=[]
    def forward(s,x):
        x=F.gelu(s.proj(x))
        for blk in s.blocks: x=blk(x)
        return s.head(x)

# ══ ARQUITECTURAS CNN (de phase10) ═══════════════════════════════════
class ConvBlock(nn.Module):
    def __init__(s, in_c, out_c, kernel=3, pool=True, dropout=0.0):
        super().__init__(); L=[nn.Conv2d(in_c,out_c,kernel,padding=kernel//2),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True)]
        if dropout>0: L.append(nn.Dropout2d(dropout))
        if pool: L.append(nn.MaxPool2d(2))
        s.block=nn.Sequential(*L)
    def forward(s,x): return s.block(x)

class SmallCNN(nn.Module):
    def __init__(s, channels, fc_dims, dropout=0.3, n_classes=NC):
        super().__init__(); blocks=[]; in_c=3
        for out_c in channels: blocks.append(ConvBlock(in_c,out_c,pool=True,dropout=dropout*0.5)); in_c=out_c
        s.features=nn.Sequential(*blocks); spatial=32//(2**len(channels)); flat=in_c*spatial*spatial
        fc=[]; prev=flat
        for d in fc_dims: fc+=[nn.Linear(prev,d),nn.ReLU(inplace=True),nn.Dropout(dropout)]; prev=d
        fc.append(nn.Linear(prev,n_classes)); s.classifier=nn.Sequential(*fc); s._acts={}; s._hooks=[]
    def _attach(s):
        for i,blk in enumerate(s.features): s._hooks.append(blk.register_forward_hook(
            lambda mo,inp,out,i=i: s._acts.update({f"cnn_block_{i}":out.mean(dim=[2,3]).detach().cpu()})))
    def _detach(s):
        for h in s._hooks: h.remove()
        s._hooks=[]
    def forward(s,x): x=s.features(x); x=x.view(x.size(0),-1); return s.classifier(x)

class ResBlockCNN(nn.Module):
    def __init__(s,c):
        super().__init__(); s.conv1=nn.Conv2d(c,c,3,padding=1,bias=False); s.bn1=nn.BatchNorm2d(c)
        s.conv2=nn.Conv2d(c,c,3,padding=1,bias=False); s.bn2=nn.BatchNorm2d(c)
    def forward(s,x):
        r=F.relu(s.bn1(s.conv1(x))); r=s.bn2(s.conv2(r)); return F.relu(x+r)

class ResNetSmall(nn.Module):
    def __init__(s, channels=[32,64,128], n_blocks=2, dropout=0.2, n_classes=NC):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv2d(3,channels[0],3,padding=1,bias=False),nn.BatchNorm2d(channels[0]),nn.ReLU(inplace=True))
        stages=[]; in_c=channels[0]
        for out_c in channels[1:]:
            stg=[nn.Sequential(nn.Conv2d(in_c,out_c,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True))]
            for _ in range(n_blocks): stg.append(ResBlockCNN(out_c))
            stages.append(nn.Sequential(*stg)); in_c=out_c
        s.stages=nn.ModuleList(stages); s.pool=nn.AdaptiveAvgPool2d(1); s.drop=nn.Dropout(dropout)
        s.fc=nn.Linear(in_c,n_classes); s._acts={}; s._hooks=[]
    def _attach(s):
        s._hooks.append(s.stem.register_forward_hook(lambda mo,inp,out: s._acts.update({"stem":out.mean(dim=[2,3]).detach().cpu()})))
        for i,stg in enumerate(s.stages): s._hooks.append(stg.register_forward_hook(
            lambda mo,inp,out,i=i: s._acts.update({f"stage_{i}":out.mean(dim=[2,3]).detach().cpu()})))
    def _detach(s):
        for h in s._hooks: h.remove()
        s._hooks=[]
    def forward(s,x):
        x=s.stem(x)
        for stg in s.stages: x=stg(x)
        x=s.pool(x).view(x.size(0),-1); return s.fc(s.drop(x))

# ══ CONFIGS (30 MLP de phase8 + 10 CNN de phase10) ════════════════════
MLP_CONFIGS = [
 {"fam":"A_Shallow","cls":"sha","hidden":[128],"dropout":0.5,"lr":5e-3,"epochs":5},
 {"fam":"A_Shallow","cls":"sha","hidden":[256],"dropout":0.4,"lr":1e-3,"epochs":8},
 {"fam":"A_Shallow","cls":"sha","hidden":[512],"dropout":0.3,"lr":5e-4,"epochs":10},
 {"fam":"A_Shallow","cls":"sha","hidden":[256,128],"dropout":0.5,"lr":2e-3,"epochs":8},
 {"fam":"A_Shallow","cls":"sha","hidden":[512,256],"dropout":0.2,"lr":5e-4,"epochs":15},
 {"fam":"A_Shallow","cls":"sha","hidden":[1024,512],"dropout":0.4,"lr":1e-3,"epochs":12},
 {"fam":"A_Shallow","cls":"sha","hidden":[512,256,128],"dropout":0.1,"lr":3e-4,"epochs":20},
 {"fam":"A_Shallow","cls":"sha","hidden":[1024,512,256],"dropout":0.3,"lr":8e-4,"epochs":15},
 {"fam":"A_Shallow","cls":"sha","hidden":[2048,1024],"dropout":0.5,"lr":2e-3,"epochs":10},
 {"fam":"A_Shallow","cls":"sha","hidden":[1024,1024,512],"dropout":0.2,"lr":1e-4,"epochs":25},
 {"fam":"B_Deep","cls":"dep","hidden":[256,128,64],"dropout":0.6,"lr":5e-3,"epochs":6},
 {"fam":"B_Deep","cls":"dep","hidden":[512,256,128],"dropout":0.4,"lr":1e-3,"epochs":10},
 {"fam":"B_Deep","cls":"dep","hidden":[512,256,128,64],"dropout":0.5,"lr":2e-3,"epochs":12},
 {"fam":"B_Deep","cls":"dep","hidden":[1024,512,256,128],"dropout":0.3,"lr":5e-4,"epochs":15},
 {"fam":"B_Deep","cls":"dep","hidden":[1024,512,256,128,64],"dropout":0.2,"lr":3e-4,"epochs":18},
 {"fam":"B_Deep","cls":"dep","hidden":[2048,1024,512,256],"dropout":0.4,"lr":1e-3,"epochs":12},
 {"fam":"B_Deep","cls":"dep","hidden":[2048,1024,512,256,128],"dropout":0.5,"lr":5e-4,"epochs":15},
 {"fam":"B_Deep","cls":"dep","hidden":[1024,1024,512,256,128],"dropout":0.1,"lr":1e-4,"epochs":25},
 {"fam":"B_Deep","cls":"dep","hidden":[4096,2048,1024,512],"dropout":0.6,"lr":2e-3,"epochs":10},
 {"fam":"B_Deep","cls":"dep","hidden":[2048,2048,1024,512,256],"dropout":0.3,"lr":4e-4,"epochs":20},
 {"fam":"C_Residual","cls":"res","hidden_dim":128,"n_blocks":2,"dropout":0.5,"lr":5e-3,"epochs":8},
 {"fam":"C_Residual","cls":"res","hidden_dim":256,"n_blocks":3,"dropout":0.4,"lr":1e-3,"epochs":12},
 {"fam":"C_Residual","cls":"res","hidden_dim":512,"n_blocks":2,"dropout":0.3,"lr":5e-4,"epochs":15},
 {"fam":"C_Residual","cls":"res","hidden_dim":512,"n_blocks":4,"dropout":0.5,"lr":2e-3,"epochs":10},
 {"fam":"C_Residual","cls":"res","hidden_dim":512,"n_blocks":6,"dropout":0.2,"lr":1e-4,"epochs":25},
 {"fam":"C_Residual","cls":"res","hidden_dim":1024,"n_blocks":3,"dropout":0.4,"lr":8e-4,"epochs":15},
 {"fam":"C_Residual","cls":"res","hidden_dim":1024,"n_blocks":5,"dropout":0.6,"lr":3e-3,"epochs":12},
 {"fam":"C_Residual","cls":"res","hidden_dim":1024,"n_blocks":8,"dropout":0.3,"lr":3e-4,"epochs":20},
 {"fam":"C_Residual","cls":"res","hidden_dim":2048,"n_blocks":4,"dropout":0.5,"lr":1e-3,"epochs":15},
 {"fam":"C_Residual","cls":"res","hidden_dim":2048,"n_blocks":6,"dropout":0.2,"lr":2e-4,"epochs":20},
]
CNN_CONFIGS = [
 {"fam":"D_CNNsmall","type":"small","channels":[32,64],"fc":[256],"dropout":0.3,"lr":1e-3,"epochs":15},
 {"fam":"D_CNNsmall","type":"small","channels":[32,64,128],"fc":[256],"dropout":0.3,"lr":5e-4,"epochs":18},
 {"fam":"D_CNNsmall","type":"small","channels":[64,128],"fc":[512,256],"dropout":0.4,"lr":1e-3,"epochs":15},
 {"fam":"D_CNNsmall","type":"small","channels":[64,128,256],"fc":[512],"dropout":0.3,"lr":5e-4,"epochs":20},
 {"fam":"D_CNNsmall","type":"small","channels":[64,128,256],"fc":[512,256],"dropout":0.4,"lr":3e-4,"epochs":22},
 {"fam":"E_CNNres","type":"resnet","channels":[32,64,128],"n_blocks":1,"dropout":0.2,"lr":1e-3,"epochs":20},
 {"fam":"E_CNNres","type":"resnet","channels":[32,64,128],"n_blocks":2,"dropout":0.2,"lr":5e-4,"epochs":25},
 {"fam":"E_CNNres","type":"resnet","channels":[64,128,256],"n_blocks":2,"dropout":0.2,"lr":3e-4,"epochs":25},
 {"fam":"E_CNNres","type":"resnet","channels":[64,128,256],"n_blocks":3,"dropout":0.3,"lr":2e-4,"epochs":30},
 {"fam":"E_CNNres","type":"resnet","channels":[64,128,256,512],"n_blocks":3,"dropout":0.3,"lr":1e-4,"epochs":30},
]

def build_mlp(cfg):
    c=cfg["cls"]
    if c=="sha": return MLPShallow(cfg["hidden"],cfg["dropout"])
    if c=="dep": return MLPDeep(cfg["hidden"],cfg["dropout"])
    return MLPResidual(cfg["hidden_dim"],cfg["n_blocks"],cfg["dropout"])
def build_cnn(cfg):
    return SmallCNN(cfg["channels"],cfg["fc"],cfg["dropout"]) if cfg["type"]=="small" else ResNetSmall(cfg["channels"],cfg["n_blocks"],cfg["dropout"])

def train(model, is_mlp, cfg):
    model.to(DEVICE)
    opt=optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["epochs"]); crit=nn.CrossEntropyLoss()
    for ep in range(cfg["epochs"]):
        model.train()
        for X,y in tr_ld:
            X=X.view(X.size(0),-1).to(DEVICE) if is_mlp else X.to(DEVICE); y=y.to(DEVICE)
            opt.zero_grad(); crit(model(X),y).backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sch.step()
    return model

def evaluate_extract(model, is_mlp, n_batches=10):
    model.eval(); model._attach(); correct=total=0; all_acts={}
    with torch.no_grad():
        for i,(X,y) in enumerate(te_ld):
            Xin=X.view(X.size(0),-1).to(DEVICE) if is_mlp else X.to(DEVICE)
            out=model(Xin); pred=out.argmax(1).cpu().numpy(); yn=y.numpy()
            correct+=(pred==yn).sum(); total+=len(yn)
            if i<n_batches:
                for k,v in model._acts.items(): all_acts.setdefault(k,[]).append(v.numpy())
    model._detach()
    acts={k:np.vstack(v) for k,v in all_acts.items()}
    return correct/total*100, acts

# ══ DESCRIPTORES (de phase10) ════════════════════════════════════════
def desc_x1(act, threshold=0.40, max_n=80):
    N=min(act.shape[1],max_n); corr=np.corrcoef(act[:,:N].T); corr=np.nan_to_num(corr); np.fill_diagonal(corr,0)
    G=nx.from_numpy_array((np.abs(corr)>threshold).astype(float)); deg=np.array([d for _,d in G.degree()])
    if len(deg)==0: deg=np.array([0.])
    try: bc=np.array(list(nx.betweenness_centrality(G).values()))
    except: bc=np.zeros(len(deg))
    return np.array([nx.density(G),nx.average_clustering(G),nx.transitivity(G),deg.mean(),deg.std(),
        deg.max()/max(1,len(deg)),float(nx.number_connected_components(G))/max(G.number_of_nodes(),1),
        G.number_of_edges()/max(G.number_of_nodes(),1),bc.mean(),bc.std()],dtype=np.float64)

def desc_x2(act, n_pts=50):
    A=act[:n_pts]; nc=min(15,A.shape[1],A.shape[0]-1)
    if nc<2: return np.zeros(12)
    Ar=PCA(n_components=nc,random_state=42).fit_transform(A); Ar=(Ar-Ar.min())/(Ar.max()-Ar.min()+1e-8)
    if HAS_RIPSER:
        dgms=ripser_fn(Ar,maxdim=1)["dgms"]; h0=dgms[0]; h0=h0[h0[:,1]<np.inf]; h1=dgms[1]; h1=h1[h1[:,1]<np.inf]
    else:
        corr=np.nan_to_num(np.corrcoef(Ar.T)); dist=1-np.abs(corr); thrs=np.linspace(0,1,30); prev=Ar.shape[0]; h0p=[]
        for ki,t in enumerate(thrs[1:],1):
            cc=nx.number_connected_components(nx.from_numpy_array((dist<t).astype(int)))
            for _ in range(max(prev-cc,0)): h0p.append([thrs[ki-1],t])
            prev=cc
        h0=np.array(h0p) if h0p else np.array([[0.,0.05]]); h1=np.zeros((0,2))
    def sig6(d):
        if len(d)==0: return np.zeros(6)
        p=d[:,1]-d[:,0]; return np.array([p.mean(),p.std(),p.max(),p.sum(),float(len(p)),np.percentile(p,75)])
    return np.concatenate([sig6(h0),sig6(h1)])

def desc_x3(x1,x2):
    n1=x1/(np.linalg.norm(x1)+1e-8); n2=x2/(np.linalg.norm(x2)+1e-8); return np.concatenate([n1,n2])

# ══ BUCLE PRINCIPAL ══════════════════════════════════════════════════
ALL = [("MLP",c) for c in MLP_CONFIGS] + [("CNN",c) for c in CNN_CONFIGS]
records=[]
for idx,(kind,cfg) in enumerate(ALL):
    is_mlp = kind=="MLP"
    model = build_mlp(cfg) if is_mlp else build_cnn(cfg)
    np_ = sum(p.numel() for p in model.parameters())
    t=time.time()
    model = train(model, is_mlp, cfg)
    acc, acts = evaluate_extract(model, is_mlp, n_batches=10)
    A = acts[list(acts.keys())[-1]]                 # última capa oculta
    x1=desc_x1(A); x2=desc_x2(A); x3=desc_x3(x1,x2)
    dt=round(time.time()-t)
    records.append({"id":idx+1,"family":cfg["fam"],"arch_type":kind,"acc_global":acc,
                    "n_params":np_,"x1_g":x1,"x2_g":x2,"x3_g":x3,"cfg":cfg,"dt_s":dt})
    print(f"  [{idx+1:02d}/40] {cfg['fam']:11s} {kind} acc={acc:5.2f}%  ({dt}s)  acts={A.shape}")
    json.dump({"done":idx+1,"total":40,"pct":round((idx+1)/40*100,1),
               "elapsed_s":round(time.time()-T0),
               "completed":[{"id":r["id"],"family":r["family"],"acc":round(r["acc_global"],2)} for r in records]},
              open(PROG,"w"))

pickle.dump(records, open(OUT/"all_records_f12.pkl","wb"))
accs=np.array([r["acc_global"] for r in records])
print(f"\n  CIFAR-100 acc=[{accs.min():.2f}, {accs.max():.2f}]  rango={accs.max()-accs.min():.2f} pp")
print(f"  Registros: {OUT/'all_records_f12.pkl'}  |  Tiempo total: {round(time.time()-T0)}s")
print("="*70); print("  FASE 12 COMPLETADA"); print("="*70)
