import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
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
OUT_PNG = os.path.join(OUT_DIR, "수질예측_예시_학습결과.png")

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
nh3 = 2.0 + base
nh3 = nh3 + np.convolve(0.15 * base, np.ones(6) / 6, mode="same")
nh3 = np.clip(nh3, 0.1, None) + 0.15 * np.random.randn(N)

features = np.column_stack([DO, pH, ORP, temp, MLSS, flow, aeration, ret_ratio, load])
target = nh3.reshape(-1, 1)

n_train = int(N * 0.7)
n_val = int(N * 0.85)

fx = MinMaxScaler().fit(features[:n_train])
fy = MinMaxScaler().fit(target[:n_train])
X = fx.transform(features)
y = fy.transform(target)

WIN = 24
HORIZON = 1


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


class BiLSTMRegressor(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(self.drop(out))


model = BiLSTMRegressor(features.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

BATCH = 16
EPOCHS = 60
PATIENCE = 8
best_val = float("inf")
best_state = None
wait = 0
train_hist, val_hist = [], []

idx = np.arange(len(Xtr_t))
for epoch in range(EPOCHS):
    model.train()
    np.random.shuffle(idx)
    ep_loss = 0.0
    for s in range(0, len(idx), BATCH):
        b = idx[s:s + BATCH]
        xb, yb = Xtr_t[b], ytr_t[b]
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        ep_loss += loss.item() * len(b)
    ep_loss /= len(idx)

    model.eval()
    with torch.no_grad():
        vloss = loss_fn(model(Xval_t), yval_t).item()
    train_hist.append(ep_loss)
    val_hist.append(vloss)

    if vloss < best_val - 1e-6:
        best_val = vloss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    pred_te = model(Xte_t).cpu().numpy()

pred_inv = fy.inverse_transform(pred_te).ravel()
true_inv = fy.inverse_transform(yte).ravel()

mae = mean_absolute_error(true_inv, pred_inv)
rmse = np.sqrt(mean_squared_error(true_inv, pred_inv))
r2 = r2_score(true_inv, pred_inv)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(train_hist, label="학습 손실(Train)", color="#2563eb")
ax1.plot(val_hist, label="검증 손실(Validation)", color="#d97706")
ax1.set_title("학습 손실 곡선 (MSE)", fontsize=15, fontweight="bold")
ax1.set_xlabel("에폭(Epoch)")
ax1.set_ylabel("손실(Loss)")
ax1.legend()
ax1.grid(alpha=0.3)

show = min(200, len(true_inv))
ax2.plot(true_inv[:show], label="실측값(예시)", color="#0f172a", linewidth=1.8)
ax2.plot(pred_inv[:show], label="예측값(Bi-LSTM)", color="#dc2626",
         linewidth=1.5, linestyle="--")
ax2.set_title("NH3-N 예측 결과 (테스트 구간)", fontsize=15, fontweight="bold")
ax2.set_xlabel("시간 스텝(시간)")
ax2.set_ylabel("NH3-N 농도 (mg/L)")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

metric_txt = (f"[예시 데이터 기준]\nMAE  = {mae:.3f} mg/L\n"
              f"RMSE = {rmse:.3f} mg/L\nR²    = {r2:.3f}")
ax2.text(0.02, 0.97, metric_txt, transform=ax2.transAxes, va="top", ha="left",
         fontsize=11, bbox=dict(boxstyle="round", fc="#f1f5f9", ec="#94a3b8"))

dev_name = "CUDA(RTX 3050)" if torch.cuda.is_available() else f"CPU (현재 torch: {torch.__version__})"
fig.suptitle(
    "Bi-LSTM 수처리 수질 예측 — 예시(가상) 데이터 학습 결과   "
    "※ 실제 현장 데이터·성능 아님",
    fontsize=16, fontweight="bold")
fig.text(0.5, 0.005,
         f"학습 장치: {dev_name}  |  목표 GPU: NVIDIA GeForce RTX 3050  |  "
         f"입력 24h · 예측 +1h · hidden 64 · 2층 · batch 16  |  가상 데이터로 코드 동작 시연",
         ha="center", fontsize=10, color="#475569")

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(OUT_PNG, dpi=130)
print("Saved:", OUT_PNG)
print(f"device={device} MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} epochs={len(train_hist)}")
