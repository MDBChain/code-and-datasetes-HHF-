import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import pickle
from copy import deepcopy
import math

# =============== 参数 ===============
HIST_LEN = 24        # 历史长度（24h）
PRED_LEN = 1         # 预测下一步（1h）
STRIDE_STEPS = 1     # 每1小时起报一次
BATCH_SIZE = 64
EPOCHS = 150
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 80        # 早停

# =============== 时间特征 ===============
def build_time_features(timestamps: pd.Series):
    """
    时间特征编码：
    - month（12周期）
    - weekday（7周期）
    - hour（24周期）
    - day_of_year（365周期）
    全部采用 sin/cos 编码
    """
    ts = timestamps.dt

    month = ts.month.values
    weekday = ts.weekday.values
    hour = ts.hour.values
    day_of_year = ts.dayofyear.values

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    doy_sin = np.sin(2 * np.pi * day_of_year / 365)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365)

    features = np.stack(
        [
            month_sin, month_cos,
            weekday_sin, weekday_cos,
            hour_sin, hour_cos,
            doy_sin, doy_cos
        ],
        axis=1
    ).astype(np.float32)

    return features


class RelativePositionBias(nn.Module):
    """
    T5 风格 RPE
    """
    def __init__(self, num_buckets=32, max_distance=128, n_heads=8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def forward(self, q_len, k_len):
        device = self.relative_attention_bias.weight.device

        context = torch.arange(q_len, device=device)[:, None]
        memory = torch.arange(k_len, device=device)[None, :]
        relative_position = memory - context  # [q,k]

        rp_bucket = self._relative_position_bucket(relative_position)
        rp_bucket = rp_bucket.to(device)

        values = self.relative_attention_bias(rp_bucket)  # [q,k,h]
        return values.permute(2, 0, 1)  # [h,q,k]

    def _relative_position_bucket(self, relative_position):
        n = torch.abs(relative_position)
        max_exact = self.num_buckets // 2

        is_small = n < max_exact

        val_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6) /
            math.log(self.max_distance / max_exact)
            * (self.num_buckets - max_exact)
        ).long()
        val_large = torch.clamp(val_large, max=self.num_buckets - 1)

        buckets = torch.where(is_small, n, val_large)
        return buckets


class MultiHeadAttentionRPE(nn.Module):
    """
    自定义 MultiheadAttention，显式加入 RPE bias
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, rpe_bias):
        B, Lq, _ = Q.size()
        _, Lk, _ = K.size()

        q = self.Wq(Q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(K).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(V).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        logits = logits + rpe_bias.unsqueeze(0)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.Wo(out)

        return out


class EncoderLayerRPE(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttentionRPE(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rpe_bias):
        h = self.attn(x, x, x, rpe_bias)
        x = self.norm1(x + self.dropout(h))

        h2 = self.ffn(x)
        x = self.norm2(x + self.dropout(h2))

        return x


class PatchTST_D1(nn.Module):
    """
    保留 D+1 版本核心结构：
    - hist encoder
    - future <- history cross attention
    - gated fusion
    这里只是把 pred_len 改成 1
    """
    def __init__(self, enc_in_hist, enc_in_future,
                 hist_len=24, pred_len=1,
                 d_model=128, n_heads=8, e_layers=3,
                 patch_len=1, dropout=0.1):
        super().__init__()

        self.hist_len = hist_len
        self.pred_len = pred_len
        self.patch_len = patch_len

        assert hist_len % patch_len == 0, "hist_len must be divisible by patch_len"
        assert pred_len % patch_len == 0, "pred_len must be divisible by patch_len"

        self.n_patch_hist = hist_len // patch_len
        self.n_patch_future = pred_len // patch_len

        self.hist_proj = nn.Linear(enc_in_hist, d_model)
        self.future_proj = nn.Linear(enc_in_future, d_model)

        self.patch_embed_hist = nn.Linear(patch_len * d_model, d_model)
        self.patch_embed_future = nn.Linear(patch_len * d_model, d_model)

        self.rpe_hist = RelativePositionBias(n_heads=n_heads)
        self.rpe_fut = RelativePositionBias(n_heads=n_heads)

        self.encoder_layers = nn.ModuleList([
            EncoderLayerRPE(d_model, n_heads, dropout)
            for _ in range(e_layers)
        ])

        self.cross_attn = MultiHeadAttentionRPE(d_model, n_heads, dropout)

        self.cross_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        self.patch_proj = nn.Linear(d_model, patch_len * d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, hist_x, fut_x):
        B = hist_x.size(0)

        hist = self.hist_proj(hist_x)      # [B, hist_len, d_model]
        fut = self.future_proj(fut_x)      # [B, pred_len, d_model]

        hist = hist.view(B, self.n_patch_hist, self.patch_len, -1)
        hist = self.patch_embed_hist(hist.reshape(B, self.n_patch_hist, -1))

        fut = fut.view(B, self.n_patch_future, self.patch_len, -1)
        fut = self.patch_embed_future(fut.reshape(B, self.n_patch_future, -1))

        rpe_hist = self.rpe_hist(self.n_patch_hist, self.n_patch_hist)
        for layer in self.encoder_layers:
            hist = layer(hist, rpe_hist)

        rpe_fut = self.rpe_fut(self.n_patch_future, self.n_patch_hist)
        cross_out = self.cross_attn(fut, hist, hist, rpe_fut)

        gate = self.cross_gate(fut)
        fut = 0.5 * gate * fut + cross_out

        fut_seq = self.patch_proj(fut)
        fut_seq = fut_seq.view(B, self.n_patch_future, self.patch_len, -1)
        fut_seq = fut_seq.reshape(B, self.pred_len, -1)

        return self.output_proj(fut_seq)


def run_one_site(site_id):
    # =============== 加载数据 ===============
    df = pd.read_csv(f"predictors_zone{site_id}.csv")

    # 时间列
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)

    # =============== 特征定义（GEFCom Solar） ===============
    weather_features = [
        "VAR78", "VAR79", "VAR134", "VAR157",
        "VAR164", "VAR165", "VAR166", "VAR167",
        "VAR169", "VAR175", "VAR178", "VAR228"
    ]
    target = "POWER"

    # =============== 标准化 ===============
    scaler_x = MinMaxScaler()
    scaler_t = MinMaxScaler()

    x_scaled = scaler_x.fit_transform(df[weather_features])  # [N, Cw]
    y_scaled = df[[target]].values.astype(np.float32)        # [N, 1]
    t_scaled = scaler_t.fit_transform(
        build_time_features(df["TIMESTAMP"])
    )  # [N, Ct]

    # =============== 构建 1h 单步预测样本 ===============
    def create_st1_samples(x, y, t, timestamps,
                           hist_len=HIST_LEN, pred_len=PRED_LEN, stride_steps=STRIDE_STEPS):
        """
        每个样本：
        历史输入：历史天气 + 历史出力 + 历史时间
        未来输入：未来天气 + 未来时间
        标签：下一步出力 [1,1]
        """
        X_hist, X_fut, Ys, pred_dates = [], [], [], []
        N = len(x)

        for i in range(0, N - hist_len - pred_len + 1, stride_steps):
            # 历史
            x_hist = x[i:i + hist_len]
            y_hist = y[i:i + hist_len]
            t_hist = t[i:i + hist_len]

            # 下一步
            x_fut = x[i + hist_len:i + hist_len + pred_len]
            y_fut = y[i + hist_len:i + hist_len + pred_len]
            t_fut = t[i + hist_len:i + hist_len + pred_len]

            hist_in = np.concatenate([x_hist, y_hist, t_hist], axis=-1)
            fut_in = np.concatenate([x_fut, t_fut], axis=-1)

            X_hist.append(hist_in)
            X_fut.append(fut_in)
            Ys.append(y_fut)
            pred_dates.append(timestamps.iloc[i + hist_len])

        return (
            np.array(X_hist, dtype=np.float32),
            np.array(X_fut, dtype=np.float32),
            np.array(Ys, dtype=np.float32),
            pd.to_datetime(pred_dates)
        )

    X_hist_all, X_fut_all, Y_all, pred_dates_all = create_st1_samples(
        x_scaled, y_scaled, t_scaled, df["TIMESTAMP"]
    )

    N_samples = len(X_hist_all)
    split_idx = int(N_samples * 0.7)

    X_hist_train = X_hist_all[:split_idx]
    X_fut_train = X_fut_all[:split_idx]
    Y_train = Y_all[:split_idx]

    X_hist_val = X_hist_all[split_idx:]
    X_fut_val = X_fut_all[split_idx:]
    Y_val = Y_all[split_idx:]
    dates_val = pred_dates_all[split_idx:]

    # =============== Dataset ===============
    class PVPatchD1Dataset(Dataset):
        def __init__(self, Xh, Xf, Y):
            self.Xh = torch.tensor(Xh, dtype=torch.float32)
            self.Xf = torch.tensor(Xf, dtype=torch.float32)
            self.Y = torch.tensor(Y, dtype=torch.float32)

        def __len__(self):
            return len(self.Xh)

        def __getitem__(self, idx):
            return self.Xh[idx], self.Xf[idx], self.Y[idx]

    train_loader = DataLoader(
        PVPatchD1Dataset(X_hist_train, X_fut_train, Y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )

    # =============== 模型 / 优化器 / 损失 ===============
    Cw = x_scaled.shape[1]
    Ct = t_scaled.shape[1]
    enc_in_hist = Cw + 1 + Ct
    enc_in_future = Cw + Ct

    model = PatchTST_D1(
        enc_in_hist=enc_in_hist,
        enc_in_future=enc_in_future,
        hist_len=HIST_LEN,
        pred_len=PRED_LEN,
        d_model=128,
        n_heads=8,
        e_layers=3,
        patch_len=1,
        dropout=0.1
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # =============== 验证函数：按月评估 ===============
    def evaluate_on_months(model):
        model.eval()
        preds_list, trues_list = [], []

        with torch.no_grad():
            for Xh, Xf, Y in zip(X_hist_val, X_fut_val, Y_val):
                Xh_t = torch.tensor(Xh, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                out = model(Xh_t, Xf_t)              # [1,1,1]
                pred = out.squeeze(0).cpu().numpy()  # [1,1]
                true = Y_t.squeeze(0).cpu().numpy()  # [1,1]

                preds_list.append(pred.squeeze(-1))  # [1]
                trues_list.append(true.squeeze(-1))  # [1]

        preds_arr = np.array(preds_list)  # [N_val,1]
        trues_arr = np.array(trues_list)  # [N_val,1]

        preds_inv = preds_arr
        trues_inv = trues_arr

        dates_series = pd.Series(dates_val)
        months = dates_series.dt.to_period("M")

        results = {}
        for m in np.unique(months):
            mask = (months == m).values
            rmse = np.sqrt(np.mean((preds_inv[mask] - trues_inv[mask]) ** 2))
            acc = 1 - rmse
            results[str(m)] = acc

        mean_acc = np.mean(list(results.values()))
        return mean_acc

    # =============== 训练 + Early Stopping ===============
    best_acc = -1e9
    best_state = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for Xh_b, Xf_b, Y_b in train_loader:
            Xh_b = Xh_b.to(DEVICE)
            Xf_b = Xf_b.to(DEVICE)
            Y_b = Y_b.to(DEVICE)

            optimizer.zero_grad()
            out = model(Xh_b, Xf_b)     # [B,1,1]
            loss = criterion(out, Y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_mean_acc = evaluate_on_months(model)

        print(f"Epoch {epoch+1}/{EPOCHS}   Loss = {train_loss:.6f}   Val_mean_Acc = {val_mean_acc:.6f}")

        if val_mean_acc > best_acc:
            best_acc = val_mean_acc
            best_state = deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}, best mean acc = {best_acc:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # =============== 最终按月评估并输出指标 ===============
    model.eval()
    preds_day, trues_day = [], []

    with torch.no_grad():
        for Xh, Xf, Y in zip(X_hist_val, X_fut_val, Y_val):
            Xh_t = torch.tensor(Xh, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            out = model(Xh_t, Xf_t)                        # [1,1,1]
            pred = out.squeeze(0).squeeze(-1).cpu().numpy()  # [1]
            true = Y.squeeze(-1)                             # [1]

            preds_day.append(pred)
            trues_day.append(true)

    preds_day = np.array(preds_day)   # [N_val,1]
    trues_day = np.array(trues_day)   # [N_val,1]

    preds_inv = preds_day
    trues_inv = trues_day

    dates_test = pd.Series(dates_val)
    months = dates_test.dt.to_period("M")

    results = {}
    for m in np.unique(months):
        mask = (months == m).values
        rmse = np.sqrt(np.mean((preds_inv[mask] - trues_inv[mask]) ** 2))
        acc = 1 - rmse
        results[str(m)] = acc

    d1_df = pd.DataFrame({
        "month": list(results.keys()),
        "D+1_RMSE/Cap": list(results.values())   # 保持原保存列名格式
    })

    print("\n========== PatchTST 风格 (历史+未来条件 1h 单步预测 + 按月测试 + EarlyStopping) ==========")
    print(d1_df)

    os.makedirs("1h-321", exist_ok=True)
    d1_df.to_excel(f"1h-321/PREPatchTST_site_{site_id}_1h_autoreg_earlystop.xlsx", index=False)

    with open(f"1h-321/PatchTST_site_{site_id}_1h_data.pkl", "wb") as f:
        pickle.dump({
            "pred": preds_inv,
            "true": trues_inv,
            "date": dates_test,
            "month_metrics": results,
            "best_mean_acc": best_acc,
        }, f)


# =============== 跑所有站点 ===============
for sid in [1, 2, 3]:
    print(f"\n========== Running Zone {sid} ==========")
    run_one_site(sid)