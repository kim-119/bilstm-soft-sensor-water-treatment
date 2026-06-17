import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "results", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 2400
t = np.arange(N)
daily = np.sin(2 * np.pi * t / 24)
weekly = 0.4 * np.sin(2 * np.pi * t / (24 * 7))

DO = 4.0 + 1.2 * daily + 0.3 * np.random.randn(N)
pH = 7.0 + 0.2 * np.sin(2 * np.pi * t / 24 + 1.0) + 0.05 * np.random.randn(N)
ORP = 120 + 30 * daily + 8 * np.random.randn(N)
temp = 18 + 6 * np.sin(2 * np.pi * t / (24 * 30)) + 0.5 * np.random.randn(N)
MLSS = 3200 + 200 * weekly + 60 * np.random.randn(N)
flow = 1000 + 250 * daily + 40 * np.random.randn(N)
aeration = 50 + 12 * daily + 3 * np.random.randn(N)
ret_ratio = 0.6 + 0.05 * weekly + 0.02 * np.random.randn(N)
load = flow * (1.0 + 0.3 * daily)

base = (
    0.9 * np.maximum(0, 2.5 - DO)
    + 0.0008 * (load - load.mean())
    - 0.04 * (temp - temp.mean())
    + 0.6 * np.maximum(0, 7.0 - pH) * 3
)
nh3 = 2.0 + base + np.convolve(0.15 * base, np.ones(6) / 6, mode="same")
nh3 = np.clip(nh3, 0.1, None) + 0.15 * np.random.randn(N)

features = np.column_stack([DO, pH, ORP, temp, MLSS, flow, aeration, ret_ratio, load])
target = nh3.reshape(-1, 1)

n_train = int(N * 0.7)
n_val = int(N * 0.85)
fx = MinMaxScaler().fit(features[:n_train])
fy = MinMaxScaler().fit(target[:n_train])
X = fx.transform(features)
y = fy.transform(target)

WIN, HORIZON = 24, 1


def make_windows(X, y, win, horizon):
    xs, ys = [], []
    for i in range(len(X) - win - horizon + 1):
        xs.append(X[i:i + win])
        ys.append(y[i + win + horizon - 1])
    return np.array(xs), np.array(ys)


Xw, yw = make_windows(X, y, WIN, HORIZON)
train_end = n_train - WIN
val_end = n_val - WIN
Xtr, ytr = Xw[:train_end], yw[:train_end]
Xval, yval = Xw[train_end:val_end], yw[train_end:val_end]
Xte, yte = Xw[val_end:], yw[val_end:]

to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
Xtr_t, ytr_t = to_t(Xtr), to_t(ytr)
Xval_t, yval_t = to_t(Xval), to_t(yval)
Xte_t = to_t(Xte)
true_inv = fy.inverse_transform(yte).ravel()


class RNNRegressor(nn.Module):
    def __init__(self, n_feat, kind, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        bi = kind == "bilstm"
        if kind == "gru":
            self.rnn = nn.GRU(n_feat, hidden, layers, batch_first=True,
                              dropout=dropout, bidirectional=False)
            out_dim = hidden
        else:
            self.rnn = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                               dropout=dropout, bidirectional=bi)
            out_dim = hidden * (2 if bi else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.drop(out[:, -1, :]))


def train_nn(kind):
    torch.manual_seed(SEED)
    model = RNNRegressor(features.shape[1], kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best, best_state, wait = float("inf"), None, 0
    idx = np.arange(len(Xtr_t))
    for epoch in range(60):
        model.train()
        np.random.shuffle(idx)
        for s in range(0, len(idx), 16):
            b = idx[s:s + 16]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xval_t), yval_t).item()
        if v < best - 1e-6:
            best, wait = v, 0
            best_state = {k: val.clone() for k, val in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 8:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).cpu().numpy()
    return fy.inverse_transform(pred).ravel()


def train_linear():
    lr = LinearRegression().fit(Xtr.reshape(len(Xtr), -1), ytr.ravel())
    pred = lr.predict(Xte.reshape(len(Xte), -1)).reshape(-1, 1)
    return fy.inverse_transform(pred).ravel()


MODELS = [
    ("선형회귀", "예시결과_선형회귀.png", "#94a3b8",
     "시계열 구조를 고려하지 않는 단순 선형 기준선(baseline)",
     train_linear),
    ("LSTM", "예시결과_LSTM.png", "#0d9488",
     "단방향(과거→현재) 순서로 시간 의존성을 학습",
     lambda: train_nn("lstm")),
    ("GRU", "예시결과_GRU.png", "#d97706",
     "게이트 수가 적어 LSTM보다 가볍고 학습이 빠름",
     lambda: train_nn("gru")),
    ("Bi-LSTM", "예시결과_Bi-LSTM.png", "#dc2626",
     "정·역방향 문맥을 함께 학습(실시간 예측 시 지연 필요)",
     lambda: train_nn("bilstm")),
]

dev_name = "CUDA(RTX 3050)" if torch.cuda.is_available() else f"CPU ({torch.__version__})"

for name, fname, color, desc, fn in MODELS:
    pred = fn()
    mae = mean_absolute_error(true_inv, pred)
    rmse = np.sqrt(mean_squared_error(true_inv, pred))
    r2 = r2_score(true_inv, pred)
    print(f"{name}: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

    fig, (axt, axs) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={"width_ratios": [1.7, 1.0]})

    show = min(160, len(true_inv))
    axt.plot(true_inv[:show], color="#0f172a", linewidth=2.2, label="실측값(예시)")
    axt.plot(pred[:show], color=color, linewidth=1.6, linestyle="--",
             label=f"예측값({name})")
    axt.set_title(f"{name} — NH3-N 예측 (테스트 구간)", fontsize=15, fontweight="bold")
    axt.set_xlabel("시간 스텝(시간)")
    axt.set_ylabel("NH3-N 농도 (mg/L)")
    axt.legend(loc="upper right")
    axt.grid(alpha=0.3)
    axt.text(0.02, 0.97,
             f"[예시 데이터 기준]\nMAE  = {mae:.3f} mg/L\nRMSE = {rmse:.3f} mg/L\nR²    = {r2:.3f}",
             transform=axt.transAxes, va="top", fontsize=11,
             bbox=dict(boxstyle="round", fc="#f1f5f9", ec="#94a3b8"))

    lo = min(true_inv.min(), pred.min())
    hi = max(true_inv.max(), pred.max())
    axs.scatter(true_inv, pred, s=10, alpha=0.5, color=color)
    axs.plot([lo, hi], [lo, hi], color="#0f172a", linewidth=1.2,
             linestyle=":", label="이상적(예측=실측)")
    axs.set_title("실측값 대비 예측값 산점도", fontsize=13, fontweight="bold")
    axs.set_xlabel("실측 NH3-N (mg/L)")
    axs.set_ylabel("예측 NH3-N (mg/L)")
    axs.legend(loc="upper left", fontsize=9)
    axs.grid(alpha=0.3)

    fig.suptitle(f"Bi-LSTM Soft Sensor — {name} 예시(가상) 데이터 학습 결과   "
                 f"※ 실제 현장 데이터·성능 아님", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005,
             f"모델 특징: {desc}  |  학습 장치: {dev_name}  |  "
             f"입력 24h · 예측 +1h · hidden 64 · 2층 · batch 16",
             ha="center", fontsize=10, color="#475569")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=130)
    plt.close(fig)
    print("Saved:", out)
