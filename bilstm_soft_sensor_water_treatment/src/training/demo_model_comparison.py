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
from matplotlib.gridspec import GridSpec

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "results", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "모델별_성능비교_예시결과.png")

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
    Xtr_f = Xtr.reshape(len(Xtr), -1)
    Xte_f = Xte.reshape(len(Xte), -1)
    lr = LinearRegression().fit(Xtr_f, ytr.ravel())
    pred = lr.predict(Xte_f).reshape(-1, 1)
    return fy.inverse_transform(pred).ravel()


preds = {
    "선형회귀": train_linear(),
    "LSTM": train_nn("lstm"),
    "GRU": train_nn("gru"),
    "Bi-LSTM": train_nn("bilstm"),
}

metrics = {}
for name, p in preds.items():
    metrics[name] = (
        mean_absolute_error(true_inv, p),
        np.sqrt(mean_squared_error(true_inv, p)),
        r2_score(true_inv, p),
    )
    print(f"{name}: MAE={metrics[name][0]:.4f} RMSE={metrics[name][1]:.4f} R2={metrics[name][2]:.4f}")

COLORS = {"선형회귀": "#94a3b8", "LSTM": "#0d9488",
          "GRU": "#d97706", "Bi-LSTM": "#dc2626"}

diffs = {
    "선형회귀": "시계열 구조 미고려 · 단순 선형 기준선(baseline)",
    "LSTM": "단방향(과거→현재) 시간 의존성 학습",
    "GRU": "게이트 수가 적어 LSTM보다 경량·빠름",
    "Bi-LSTM": "정·역방향 문맥 모두 학습 / 실시간엔 지연 필요",
}

fig = plt.figure(figsize=(16, 9))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.28)

ax_pred = fig.add_subplot(gs[0, :])
show = min(160, len(true_inv))
ax_pred.plot(true_inv[:show], color="#0f172a", linewidth=2.4, label="실측값(예시)")
for name, p in preds.items():
    ax_pred.plot(p[:show], color=COLORS[name], linewidth=1.4, alpha=0.9,
                 linestyle="--", label=f"예측: {name}")
ax_pred.set_title("모델별 NH3-N 예측 비교 (테스트 구간)", fontsize=15, fontweight="bold")
ax_pred.set_xlabel("시간 스텝(시간)")
ax_pred.set_ylabel("NH3-N 농도 (mg/L)")
ax_pred.legend(ncol=5, loc="upper right", fontsize=10)
ax_pred.grid(alpha=0.3)

names = list(metrics.keys())
x = np.arange(len(names))
mae_v = [metrics[n][0] for n in names]
rmse_v = [metrics[n][1] for n in names]
r2_v = [metrics[n][2] for n in names]
bar_colors = [COLORS[n] for n in names]

ax_err = fig.add_subplot(gs[1, 0])
w = 0.38
ax_err.bar(x - w / 2, mae_v, w, label="MAE", color="#3b82f6")
ax_err.bar(x + w / 2, rmse_v, w, label="RMSE", color="#f59e0b")
ax_err.set_title("오차 비교 (낮을수록 좋음)", fontsize=13, fontweight="bold")
ax_err.set_ylabel("mg/L")
ax_err.set_xticks(x)
ax_err.set_xticklabels(names, fontsize=10)
ax_err.legend(fontsize=9)
ax_err.grid(axis="y", alpha=0.3)

ax_r2 = fig.add_subplot(gs[1, 1])
ax_r2.bar(x, r2_v, 0.55, color=bar_colors)
for i, v in enumerate(r2_v):
    ax_r2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
ax_r2.set_title("R² 비교 (높을수록 좋음)", fontsize=13, fontweight="bold")
ax_r2.set_ylim(0, 1.0)
ax_r2.set_xticks(x)
ax_r2.set_xticklabels(names, fontsize=10)
ax_r2.grid(axis="y", alpha=0.3)

ax_diff = fig.add_subplot(gs[1, 2])
ax_diff.axis("off")
ax_diff.set_title("모델별 특징(차이점)", fontsize=13, fontweight="bold", loc="left")
yy = 0.86
for name in names:
    ax_diff.plot([0.02], [yy + 0.02], marker="s", markersize=11,
                 color=COLORS[name], transform=ax_diff.transAxes)
    ax_diff.text(0.08, yy + 0.02, name, transform=ax_diff.transAxes,
                 fontsize=11, fontweight="bold", va="center")
    ax_diff.text(0.08, yy - 0.07, diffs[name], transform=ax_diff.transAxes,
                 fontsize=9.5, color="#334155", va="center")
    yy -= 0.24

dev_name = "CUDA(RTX 3050)" if torch.cuda.is_available() else f"CPU ({torch.__version__})"
fig.suptitle("Bi-LSTM vs 비교 모델 — 예시(가상) 데이터 학습 결과   ※ 실제 현장 데이터·성능 아님",
             fontsize=17, fontweight="bold")
fig.text(0.5, 0.01,
         f"학습 장치: {dev_name}  |  목표 GPU: NVIDIA GeForce RTX 3050  |  "
         f"입력 24h · 예측 +1h · hidden 64 · 2층 · batch 16  |  가상 데이터로 모델 비교 시연(상대 비교용)",
         ha="center", fontsize=10, color="#475569")

plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print("Saved:", OUT_PNG)
