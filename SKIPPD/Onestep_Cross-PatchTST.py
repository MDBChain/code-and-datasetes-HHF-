import os
import math
import pickle
from copy import deepcopy

import h5py
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

import torchvision.models as models


# =======================
# 0) 全局参数
# =======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 数据路径 ---
H5_PATH = "2017_2019_images_pv_processed.hdf5"
TIMES_PATH = "times_trainval.npy"
GROUP = "trainval"

# --- 时间范围 ---
START_DATE = "2019-05-20"
END_DATE = "2019-10-26"

# --- 数据组织 ---
FREQ = "15min"
DAY_LEN = 54  # 每天固定54步，不足补0，超过截断

# --- 15min一步预测 ---
HIST_LEN = 54
PRED_LEN = 1
STRIDE_STEPS = 1
PATCH_LEN = 1

# --- 训练参数 ---
BATCH_SIZE = 16
EPOCHS = 100
LR = 4e-4
PATIENCE = 40

# --- 指标中的容量 ---
capacity = 30.1


# =======================
# 1) 云图物理特征 + gate
# =======================
class PhysicsGuidedGate(nn.Module):
    def __init__(self, phys_dim, img_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phys_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, img_dim),
            nn.Sigmoid()
        )

    def forward(self, img_feat, phys_feat):
        gate = self.net(phys_feat)
        return img_feat * gate


class CloudPhysicsExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 4

    @torch.no_grad()
    def forward(self, x):
        """
        x: [B,L,3,H,W]
        return: [B,L,4]
        """
        if x.dim() != 5:
            raise ValueError(f"CloudPhysicsExtractor expects 5D input, got shape={x.shape}")

        B, L = x.shape[:2]
        x = x.reshape(B * L, 3, x.shape[-2], x.shape[-1])

        x_np = x.detach().cpu().numpy()
        feats = []

        for img in x_np:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_u8.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

            mean_brightness = gray.mean() / 255.0
            std_brightness = gray.std() / 255.0
            thresh = gray.mean()
            cloud_frac = (gray > thresh).mean()
            edges = cv2.Canny(gray, 50, 150)
            edge_density = edges.mean() / 255.0

            feats.append([
                mean_brightness,
                std_brightness,
                cloud_frac,
                edge_density
            ])

        feats = torch.tensor(feats, dtype=torch.float32, device=x.device)
        feats = feats.view(B, L, -1)
        return feats


# =======================
# 2) 时间特征
# =======================
def build_time_features(timestamps: pd.Series):
    ts = timestamps.dt

    month = ts.month.values
    weekday = ts.weekday.values
    hour = ts.hour.values
    minute = ts.minute.values
    day_of_year = ts.dayofyear.values

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    minute_slot = minute / 60.0
    minute_sin = np.sin(2 * np.pi * minute_slot)
    minute_cos = np.cos(2 * np.pi * minute_slot)

    doy_sin = np.sin(2 * np.pi * day_of_year / 365)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365)

    features = np.stack(
        [
            month_sin, month_cos,
            weekday_sin, weekday_cos,
            hour_sin, hour_cos,
            minute_sin, minute_cos,
            doy_sin, doy_cos
        ],
        axis=1
    ).astype(np.float32)

    return features


# =======================
# 3) PatchTST
# =======================
class RelativePositionBias(nn.Module):
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
        relative_position = memory - context
        rp_bucket = self._relative_position_bucket(relative_position).to(device)
        values = self.relative_attention_bias(rp_bucket)
        return values.permute(2, 0, 1)

    def _relative_position_bucket(self, relative_position):
        n = torch.abs(relative_position)
        max_exact = self.num_buckets // 2
        is_small = n < max_exact

        val_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6)
            / math.log(self.max_distance / max_exact)
            * (self.num_buckets - max_exact)
        ).long()
        val_large = torch.clamp(val_large, max=self.num_buckets - 1)

        buckets = torch.where(is_small, n, val_large)
        return buckets


class MultiHeadAttentionRPE(nn.Module):
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


class PatchTST_ST15(nn.Module):
    def __init__(self, enc_in_hist, enc_in_future,
                 hist_len=54, pred_len=1,
                 d_model=128, n_heads=8, e_layers=3,
                 patch_len=1, dropout=0.1):
        super().__init__()

        assert hist_len % patch_len == 0
        assert pred_len % patch_len == 0

        self.hist_len = hist_len
        self.pred_len = pred_len
        self.patch_len = patch_len

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

        hist = self.hist_proj(hist_x)
        fut = self.future_proj(fut_x)

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


# =======================
# 4) CNN + PatchTST
# =======================
class VideoCNN(nn.Module):
    def __init__(self, img_dim=128, pretrained=True, ksize=3):
        super().__init__()

        base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        old_conv = base.conv1
        base.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():
            base.conv1.weight[:, :3] = old_conv.weight
            base.conv1.weight[:, 3:] = 0.0

        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(512, img_dim)

        self.temporal_conv = nn.Conv1d(
            in_channels=img_dim,
            out_channels=img_dim,
            kernel_size=ksize,
            padding=ksize // 2
        )

        # 单步场景下比 BatchNorm 更稳
        self.norm = nn.GroupNorm(1, img_dim)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x: [B,L,3,64,64]
        return: [B,L,img_dim]
        """
        if x.dim() != 5:
            raise ValueError(f"VideoCNN expects 5D input, got shape={x.shape}")

        B, L, C, H, W = x.shape
        if C != 3:
            raise ValueError(f"VideoCNN expects channel=3 before concat, got {C}")

        # 关键修复：L=1 时不能做帧差，直接补零差分
        if L == 1:
            delta = torch.zeros_like(x)
        else:
            delta = x[:, 1:] - x[:, :-1]          # [B,L-1,3,H,W]
            zero = torch.zeros_like(x[:, :1])     # [B,1,3,H,W]
            delta = torch.cat([zero, delta], dim=1)

        # [B,L,6,H,W]
        x = torch.cat([x, delta], dim=2)

        x = x.reshape(B * L, 6, H, W)
        feat = self.backbone(x)
        feat = self.proj(feat)

        feat = feat.reshape(B, L, -1)          # [B,L,img_dim]
        feat = feat.transpose(1, 2)            # [B,img_dim,L]
        feat = self.temporal_conv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = feat.transpose(1, 2)            # [B,L,img_dim]

        return feat


class PatchTSTWithCNN(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        ct_dim,
        img_dim=128,
        patchtst_d_model=256,
        patchtst_heads=8,
        patchtst_layers=3,
        patch_len=1,
        dropout=0.1,
        cnn_pretrained=True,
    ):
        super().__init__()
        self.cnn = VideoCNN(
            img_dim=img_dim,
            pretrained=cnn_pretrained,
            ksize=3
        )

        self.cloud_phys = CloudPhysicsExtractor()
        self.phys_dim = 4
        self.phys_gate = PhysicsGuidedGate(
            phys_dim=self.phys_dim,
            img_dim=img_dim,
            hidden_dim=64
        )

        enc_in_hist = img_dim + 1 + ct_dim + self.phys_dim
        enc_in_fut = img_dim + ct_dim + self.phys_dim

        self.patchtst = PatchTST_ST15(
            enc_in_hist=enc_in_hist,
            enc_in_future=enc_in_fut,
            hist_len=hist_len,
            pred_len=pred_len,
            d_model=patchtst_d_model,
            n_heads=patchtst_heads,
            e_layers=patchtst_layers,
            patch_len=patch_len,
            dropout=dropout
        )

    def forward(self, hist_img, y_hist, t_hist, fut_img, t_fut):
        hist_feat = self.cnn(hist_img)
        fut_feat = self.cnn(fut_img)

        hist_phys = self.cloud_phys(hist_img)
        fut_phys = self.cloud_phys(fut_img)

        hist_feat = self.phys_gate(hist_feat, hist_phys)
        fut_feat = self.phys_gate(fut_feat, fut_phys)

        hist_in = torch.cat([hist_feat, hist_phys, y_hist, t_hist], dim=-1)
        fut_in = torch.cat([fut_feat, fut_phys, t_fut], dim=-1)

        return self.patchtst(hist_in, fut_in)


# =======================
# 5) 按天整理成54步：不足补0，超过截断
# =======================
def load_skippd_daily54_window(h5_path, times_path, group, start_date, end_date, freq="15min", day_len=54):
    times = pd.to_datetime(np.load(times_path, allow_pickle=True))
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    with h5py.File(h5_path, "r") as f:
        imgs_ds = f[f"{group}/images_log"]
        pv_ds = f[f"{group}/pv_log"]

        df = pd.DataFrame({
            "time": times,
            "idx": np.arange(len(times))
        })
        df = df[
            (df["time"] >= start_date) &
            (df["time"] <= end_date + pd.Timedelta(days=1) - pd.Timedelta(minutes=1))
        ].copy()
        df = df.sort_values("time").reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("指定时间窗口内没有数据。")

        df["date"] = df["time"].dt.floor("D")
        day_blocks = []

        for day, g in df.groupby("date"):
            g = g.sort_values("time").reset_index(drop=True)

            sub = g.set_index("time")
            idx_resampled = sub["idx"].resample(freq).first()

            times_day = idx_resampled.index
            valid_mask = idx_resampled.notna().values

            T = len(times_day)
            pv_day = np.zeros((T, 1), dtype=np.float32)
            img_day = np.zeros((T, 3, 64, 64), dtype=np.uint8)

            valid_idx = idx_resampled.dropna().astype(int).values
            valid_pos = np.where(valid_mask)[0]

            if len(valid_idx) > 0:
                pv_valid = pv_ds[valid_idx].astype(np.float32).reshape(-1, 1)
                img_valid = imgs_ds[valid_idx]
                img_valid = np.transpose(img_valid, (0, 3, 1, 2))
                pv_day[valid_pos] = pv_valid
                img_day[valid_pos] = img_valid

            if T >= day_len:
                times_fix = times_day[:day_len]
                pv_fix = pv_day[:day_len]
                img_fix = img_day[:day_len]
                mask_fix = valid_mask[:day_len]
            else:
                pad_n = day_len - T
                if T == 0:
                    continue

                pad_times = pd.date_range(
                    start=times_day[-1] + pd.Timedelta(minutes=15),
                    periods=pad_n,
                    freq=freq
                )
                times_fix = times_day.append(pad_times)
                pv_fix = np.concatenate([pv_day, np.zeros((pad_n, 1), dtype=np.float32)], axis=0)
                img_fix = np.concatenate([img_day, np.zeros((pad_n, 3, 64, 64), dtype=np.uint8)], axis=0)
                mask_fix = np.concatenate([valid_mask, np.zeros(pad_n, dtype=bool)], axis=0)

            day_blocks.append({
                "date": day,
                "times": pd.DatetimeIndex(times_fix),
                "pv": pv_fix.astype(np.float32),
                "img": img_fix.astype(np.uint8),
                "mask": mask_fix.astype(bool)
            })

    if len(day_blocks) == 0:
        raise ValueError("没有生成任何日块。")

    return day_blocks


# =======================
# 6) 构造15min一步预测样本
#    历史可跨天，预测点必须落在当天内部
# =======================
def create_st15_samples_from_day54(day_blocks, scaler_y, scaler_t,
                                   hist_len=54, pred_len=1, stride_steps=1):
    times_all = pd.DatetimeIndex(np.concatenate([b["times"].values for b in day_blocks]))
    pv_all = np.concatenate([b["pv"] for b in day_blocks], axis=0)
    img_all = np.concatenate([b["img"] for b in day_blocks], axis=0)
    mask_all = np.concatenate([b["mask"] for b in day_blocks], axis=0)

    time_features = build_time_features(pd.Series(times_all))
    time_features_s = scaler_t.transform(time_features)
    pv_s = scaler_y.transform(pv_all)

    X_hist_img, Y_hist, T_hist = [], [], []
    X_fut_img, T_fut, Y_fut, M_fut = [], [], [], []
    pred_times = []

    n_days = len(day_blocks)

    for d in range(n_days):
        day_start = d * DAY_LEN

        for s in range(0, DAY_LEN, stride_steps):
            abs_pred_start = day_start + s
            abs_pred_end = abs_pred_start + pred_len

            hist_start = abs_pred_start - hist_len
            hist_end = abs_pred_start

            if hist_start < 0:
                continue

            if s + pred_len > DAY_LEN:
                continue

            hist_img = img_all[hist_start:hist_end]
            hist_pv = pv_s[hist_start:hist_end]
            hist_tf = time_features_s[hist_start:hist_end]

            fut_img_ = img_all[abs_pred_start:abs_pred_end]
            fut_tf_ = time_features_s[abs_pred_start:abs_pred_end]
            fut_pv_ = pv_s[abs_pred_start:abs_pred_end]
            fut_m_ = mask_all[abs_pred_start:abs_pred_end]
            pred_t_ = times_all[abs_pred_start:abs_pred_end]

            # 形状兜底，避免意外空切片
            if len(hist_img) != hist_len:
                continue
            if len(fut_img_) != pred_len:
                continue

            if fut_m_.sum() <= 0:
                continue

            X_hist_img.append(hist_img.astype(np.uint8))
            Y_hist.append(hist_pv.astype(np.float32))
            T_hist.append(hist_tf.astype(np.float32))

            X_fut_img.append(fut_img_.astype(np.uint8))
            T_fut.append(fut_tf_.astype(np.float32))
            Y_fut.append(fut_pv_.astype(np.float32))
            M_fut.append(fut_m_.astype(bool))
            pred_times.append(pd.to_datetime(pred_t_))

    if len(X_hist_img) == 0:
        raise ValueError("未生成任何15min样本，请检查 HIST_LEN / 时间窗口。")

    return (
        np.stack(X_hist_img),
        np.stack(Y_hist),
        np.stack(T_hist),
        np.stack(X_fut_img),
        np.stack(T_fut),
        np.stack(Y_fut),
        np.stack(M_fut),
        np.stack(pred_times)
    )


# =======================
# 7) Dataset + masked loss
# =======================
class SKIPPDST15Dataset(Dataset):
    def __init__(self, hist_img, y_hist, t_hist, fut_img, t_fut, y_fut, m_fut):
        self.hist_img = torch.from_numpy(hist_img)
        self.y_hist = torch.from_numpy(y_hist)
        self.t_hist = torch.from_numpy(t_hist)
        self.fut_img = torch.from_numpy(fut_img)
        self.t_fut = torch.from_numpy(t_fut)
        self.y_fut = torch.from_numpy(y_fut)
        self.m_fut = torch.from_numpy(m_fut)

    def __len__(self):
        return self.hist_img.shape[0]

    def __getitem__(self, idx):
        return (
            self.hist_img[idx],
            self.y_hist[idx],
            self.t_hist[idx],
            self.fut_img[idx],
            self.t_fut[idx],
            self.y_fut[idx],
            self.m_fut[idx]
        )


def masked_mse(pred, target, mask):
    mask_f = mask.unsqueeze(-1).float()
    diff2 = (pred - target) ** 2 * mask_f
    denom = mask_f.sum().clamp_min(1.0)
    return diff2.sum() / denom


# =======================
# 8) 主流程
# =======================
def main():
    day_blocks = load_skippd_daily54_window(
        h5_path=H5_PATH,
        times_path=TIMES_PATH,
        group=GROUP,
        start_date=START_DATE,
        end_date=END_DATE,
        freq=FREQ,
        day_len=DAY_LEN
    )
    print(f"有效天数: {len(day_blocks)}")

    times_all = pd.DatetimeIndex(np.concatenate([b["times"].values for b in day_blocks]))
    pv_all = np.concatenate([b["pv"] for b in day_blocks], axis=0)
    mask_all = np.concatenate([b["mask"] for b in day_blocks], axis=0)

    scaler_y = MinMaxScaler()
    scaler_t = MinMaxScaler()

    scaler_y.fit(pv_all[mask_all].reshape(-1, 1))
    all_tf = build_time_features(pd.Series(times_all))
    scaler_t.fit(all_tf)

    Ct = all_tf.shape[1]

    (Xh_img, Yh, Th,
     Xf_img, Tf, Yf, Mf, pred_times_all) = create_st15_samples_from_day54(
        day_blocks=day_blocks,
        scaler_y=scaler_y,
        scaler_t=scaler_t,
        hist_len=HIST_LEN,
        pred_len=PRED_LEN,
        stride_steps=STRIDE_STEPS
    )

    N_samples = Xh_img.shape[0]
    print(f"15min样本数: {N_samples}")

    split_idx = int(N_samples * 0.7)

    train_set = SKIPPDST15Dataset(
        Xh_img[:split_idx], Yh[:split_idx], Th[:split_idx],
        Xf_img[:split_idx], Tf[:split_idx], Yf[:split_idx], Mf[:split_idx]
    )
    val_set = SKIPPDST15Dataset(
        Xh_img[split_idx:], Yh[split_idx:], Th[split_idx:],
        Xf_img[split_idx:], Tf[split_idx:], Yf[split_idx:], Mf[split_idx:]
    )
    pred_times_val = pred_times_all[split_idx:]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    model = PatchTSTWithCNN(
        hist_len=HIST_LEN,
        pred_len=PRED_LEN,
        ct_dim=Ct,
        img_dim=128,
        patchtst_d_model=256,
        patchtst_heads=8,
        patchtst_layers=3,
        patch_len=PATCH_LEN,
        dropout=0.1,
        cnn_pretrained=True
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def evaluate_monthly_accuracy(model_):
        model_.eval()

        preds_list, trues_list, masks_list, times_list = [], [], [], []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                hist_img, y_hist, t_hist, fut_img, t_fut, y_fut, m_fut = batch

                hist_img = hist_img.to(DEVICE).float() / 255.0
                fut_img = fut_img.to(DEVICE).float() / 255.0
                y_hist = y_hist.to(DEVICE)
                t_hist = t_hist.to(DEVICE)
                t_fut = t_fut.to(DEVICE)
                y_fut = y_fut.to(DEVICE)
                m_fut = m_fut.to(DEVICE)

                out = model_(hist_img, y_hist, t_hist, fut_img, t_fut)

                pred = out.squeeze(0).cpu().numpy().reshape(-1)
                true = y_fut.squeeze(0).cpu().numpy().reshape(-1)
                mask = m_fut.squeeze(0).cpu().numpy().astype(bool)
                pt = np.array(pred_times_val[i], dtype="datetime64[ns]").reshape(-1)

                preds_list.append(pred)
                trues_list.append(true)
                masks_list.append(mask)
                times_list.append(pt)

        preds_arr = np.concatenate(preds_list, axis=0)
        trues_arr = np.concatenate(trues_list, axis=0)
        masks_arr = np.concatenate(masks_list, axis=0)
        times_arr = pd.to_datetime(np.concatenate(times_list, axis=0))

        preds_inv = scaler_y.inverse_transform(preds_arr.reshape(-1, 1)).reshape(-1)
        trues_inv = scaler_y.inverse_transform(trues_arr.reshape(-1, 1)).reshape(-1)

        months = pd.Series(times_arr).dt.to_period("M")

        metrics = {}
        rmse_acc_list = []

        for m in np.unique(months):
            mask_m = (months == m).values & masks_arr
            if mask_m.sum() == 0:
                continue

            pred_m = preds_inv[mask_m]
            true_m = trues_inv[mask_m]

            mae = np.mean(np.abs(pred_m - true_m))
            rmse = np.sqrt(np.mean((pred_m - true_m) ** 2))

            acc_mae = 1 - mae / capacity
            acc_rmse = 1 - rmse / capacity

            metrics[str(m)] = {
                "1-MAE/Cap": acc_mae,
                "1-RMSE/Cap": acc_rmse
            }
            rmse_acc_list.append(acc_rmse)

        mean_rmse_acc = float(np.mean(rmse_acc_list)) if len(rmse_acc_list) else -1e9
        return mean_rmse_acc, metrics

    best_acc = -1e9
    best_state = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            hist_img, y_hist, t_hist, fut_img, t_fut, y_fut, m_fut = batch

            hist_img = hist_img.to(DEVICE).float() / 255.0
            fut_img = fut_img.to(DEVICE).float() / 255.0
            y_hist = y_hist.to(DEVICE)
            t_hist = t_hist.to(DEVICE)
            t_fut = t_fut.to(DEVICE)
            y_fut = y_fut.to(DEVICE)
            m_fut = m_fut.to(DEVICE)

            optimizer.zero_grad()
            out = model(hist_img, y_hist, t_hist, fut_img, t_fut)
            loss = masked_mse(out, y_fut, m_fut)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_mean_acc, _ = evaluate_monthly_accuracy(model)

        print(f"Epoch {epoch + 1}/{EPOCHS}   Loss = {train_loss:.6f}   Val_mean_1-RMSE/Cap = {val_mean_acc:.6f}")

        if val_mean_acc > best_acc:
            best_acc = val_mean_acc
            best_state = deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}, best mean acc = {best_acc:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds_all, trues_all, masks_all_out, times_all_out = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            hist_img, y_hist, t_hist, fut_img, t_fut, y_fut, m_fut = batch

            hist_img = hist_img.to(DEVICE).float() / 255.0
            fut_img = fut_img.to(DEVICE).float() / 255.0
            y_hist = y_hist.to(DEVICE)
            t_hist = t_hist.to(DEVICE)
            t_fut = t_fut.to(DEVICE)
            y_fut = y_fut.to(DEVICE)
            m_fut = m_fut.to(DEVICE)

            out = model(hist_img, y_hist, t_hist, fut_img, t_fut)

            pred = out.squeeze(0).cpu().numpy().reshape(-1)
            true = y_fut.squeeze(0).cpu().numpy().reshape(-1)
            mk = m_fut.squeeze(0).cpu().numpy().astype(bool)
            pt = np.array(pred_times_val[i], dtype="datetime64[ns]").reshape(-1)

            preds_all.append(pred)
            trues_all.append(true)
            masks_all_out.append(mk)
            times_all_out.append(pt)

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)
    masks_all_out = np.concatenate(masks_all_out, axis=0)
    times_all_out = pd.to_datetime(np.concatenate(times_all_out, axis=0))

    preds_inv = scaler_y.inverse_transform(preds_all.reshape(-1, 1)).reshape(-1)
    trues_inv = scaler_y.inverse_transform(trues_all.reshape(-1, 1)).reshape(-1)

    months = pd.Series(times_all_out).dt.to_period("M")

    results = {}
    for m in np.unique(months):
        mask_m = (months == m).values & masks_all_out
        if mask_m.sum() == 0:
            continue

        pred_m = preds_inv[mask_m]
        true_m = trues_inv[mask_m]

        mae = np.mean(np.abs(pred_m - true_m))
        rmse = np.sqrt(np.mean((pred_m - true_m) ** 2))

        acc_mae = 1 - mae / capacity
        acc_rmse = 1 - rmse / capacity

        results[str(m)] = {
            "1-MAE/Cap": acc_mae,
            "1-RMSE/Cap": acc_rmse
        }

    result_df = pd.DataFrame({
        "month": list(results.keys()),
        "1-MAE/Cap": [results[k]["1-MAE/Cap"] for k in results.keys()],
        "1-RMSE/Cap": [results[k]["1-RMSE/Cap"] for k in results.keys()]
    })

    print("\n========== SKIPPD + 每天54步补零截断 + 15min一步预测 ==========")
    print(result_df)

    save_dir = "st15_PatchTST_from_day54"
    os.makedirs(save_dir, exist_ok=True)

    out_xlsx = f"{save_dir}/SKIPPD_PatchTST_st15_{START_DATE}_{END_DATE}.xlsx"
    result_df.to_excel(out_xlsx, index=False)

    with open(f"{save_dir}/SKIPPD_PatchTST_st15_{START_DATE}_{END_DATE}.pkl", "wb") as f:
        pickle.dump({
            "pred": preds_inv,
            "true": trues_inv,
            "mask": masks_all_out,
            "pred_time": pd.Series(times_all_out),
            "month_metrics": results,
            "best_mean_1-RMSE/Cap": best_acc,
            "cfg": {
                "DAY_LEN": DAY_LEN,
                "HIST_LEN": HIST_LEN,
                "PRED_LEN": PRED_LEN,
                "STRIDE_STEPS": STRIDE_STEPS,
                "PATCH_LEN": PATCH_LEN,
                "START_DATE": START_DATE,
                "END_DATE": END_DATE
            }
        }, f)

    print(f"\n已保存：{out_xlsx}")


if __name__ == "__main__":
    main()