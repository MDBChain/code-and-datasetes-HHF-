# SKIPPD_D1_PatchTST_varlen.py
import os
import math
import pickle
from copy import deepcopy

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

import torchvision.models as models

# =======================
# 0) 全局参数（按你的实验需求改）
# =======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 数据路径（你自己改成实际路径） ---
H5_PATH = "2017_2019_images_pv_processed.hdf5"
TIMES_PATH = "times_trainval.npy"   # 这里用 trainval 的时间轴（你也可以换成合并后的）
GROUP = "trainval"                  # 用 trainval 部分作为数据源（你已选定自己的时间段）

# --- 你选定的时间段（已确认） ---
START_DATE = "2019-05-20"
END_DATE   = "2019-10-26"

# --- 粒度与序列长度 ---
FREQ = "15min"
# PatchTST 需要固定长度且能整除 patch_len。你原代码 patch_len=6，
# 54 可整除 6，并且覆盖绝大多数天（你统计 max~56）。
SEQ_LEN = 54
PATCH_LEN = 6

# --- D+1 设定 ---
HIST_DAYS = 1  # D天历史（你现在主要是前一天→后一天）；若要多天，把这里改大即可
PRED_DAYS = 1  # D+1 预测 1 天

# --- 训练参数 ---
BATCH_SIZE = 16
EPOCHS = 100
LR = 4e-4
PATIENCE = 40

# --- 指标中的容量（保持你原口径）---
capacity = 30.1

import cv2
class PhysicsGuidedGate(nn.Module):
    """
    用云物理特征生成 gate，调制图像特征
    输入:
      img_feat  : [B,L,img_dim]
      phys_feat : [B,L,phys_dim]
    输出:
      gated_img_feat : [B,L,img_dim]
    """
    def __init__(self, phys_dim, img_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phys_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, img_dim),
            nn.Sigmoid()
        )

    def forward(self, img_feat, phys_feat):
        gate = self.net(phys_feat)        # [B,L,img_dim]
        return img_feat * gate

class CloudPhysicsExtractor(nn.Module):
    """
    从单帧云图中提取物理特征
    输入:  [B,L,3,H,W] 或 [B*L,3,H,W]
    输出:  [B,L,P]
    """
    def __init__(self):
        super().__init__()
        self.out_dim = 4  # 特征维度固定

    @torch.no_grad()
    def forward(self, x):
        """
        x: [B,L,3,H,W]
        """
        if x.dim() == 5:
            B, L = x.shape[:2]
            x = x.reshape(B * L, 3, x.shape[-2], x.shape[-1])
        else:
            B = None

        x_np = x.detach().cpu().numpy()
        feats = []

        for img in x_np:
            # img: [3,H,W]
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_u8.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

            mean_brightness = gray.mean() / 255.0
            std_brightness  = gray.std() / 255.0

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

        if B is not None:
            feats = feats.view(B, L, -1)

        return feats

# =======================
# 1) 时间特征（保留你原来的 sin/cos 思路）
# =======================
def build_time_features(timestamps: pd.Series):
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
        [month_sin, month_cos,
         weekday_sin, weekday_cos,
         hour_sin, hour_cos,
         doy_sin, doy_cos],
        axis=1
    ).astype(np.float32)
    return features


# =======================
# 2) PatchTST
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
        memory  = torch.arange(k_len, device=device)[None, :]
        relative_position = memory - context  # [q,k]
        rp_bucket = self._relative_position_bucket(relative_position).to(device)
        values = self.relative_attention_bias(rp_bucket)  # [q,k,h]
        return values.permute(2, 0, 1)                    # [h,q,k]

    def _relative_position_bucket(self, relative_position):
        n = torch.abs(relative_position)
        max_exact = self.num_buckets // 2
        is_small = n < max_exact

        val_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6) /
            math.log(self.max_distance / max_exact) *
            (self.num_buckets - max_exact)
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


class PatchTST_D1(nn.Module):
    def __init__(self, enc_in_hist, enc_in_future,
                 hist_len=96, pred_len=96,
                 d_model=128, n_heads=8, e_layers=3,
                 patch_len=12, dropout=0.1):
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
        self.rpe_fut  = RelativePositionBias(n_heads=n_heads)

        self.encoder_layers = nn.ModuleList([
            EncoderLayerRPE(d_model, n_heads, dropout)
            for _ in range(e_layers)
        ])
        self.cross_attn = MultiHeadAttentionRPE(d_model, n_heads, dropout)

        # ===== 新增：gated cross-attn（弱化 fut）=====
        # 给每个 future patch 一个 gate：0~1，决定 cross-attn 修正强度
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
        fut  = self.future_proj(fut_x)

        hist = hist.view(B, self.n_patch_hist, self.patch_len, -1)
        hist = self.patch_embed_hist(hist.reshape(B, self.n_patch_hist, -1))

        fut = fut.view(B, self.n_patch_future, self.patch_len, -1)
        fut = self.patch_embed_future(fut.reshape(B, self.n_patch_future, -1))

        rpe_hist = self.rpe_hist(self.n_patch_hist, self.n_patch_hist)
        for layer in self.encoder_layers:
            hist = layer(hist, rpe_hist)

        rpe_fut = self.rpe_fut(self.n_patch_future, self.n_patch_hist)

        # cross-attn 修正项
        cross_out = self.cross_attn(fut, hist, hist, rpe_fut)  # [B, n_patch_future, d_model]

        # ===== 新增：gate（逐 future patch）=====
        # gate: [B, n_patch_future, 1]，每个 future patch 一个强度
        gate = self.cross_gate(fut)

        # ===== gated residual：保留 fut，自适应加入修正 =====
        fut = 0.5*gate *fut +  cross_out

        fut_seq = self.patch_proj(fut)
        fut_seq = fut_seq.view(B, self.n_patch_future, self.patch_len, -1)
        fut_seq = fut_seq.reshape(B, self.pred_len, -1)

        return self.output_proj(fut_seq)


# =======================
# 3) 你要求的：CNN + TransformerEncoder + AttentionPooling，再到 PatchTST
#    注意：PatchTST 不改，只在前面做“图像->特征”的封装
# =======================
class VideoCNN(nn.Module):
    """
    Frame-difference + ResNet18
    输入:  [B,L,3,64,64]
    内部:  拼成 [B,L,6,64,64] = RGB + ΔRGB
    输出:  [B,L,img_dim]
    """
    def __init__(self, img_dim=128, pretrained=True, ksize=3):
        super().__init__()

        # -------- ResNet18 backbone --------
        base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # ===== 改第一层：3通道 → 6通道 =====
        old_conv = base.conv1  # Conv2d(3,64,...)
        base.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # ===== 关键：初始化 6 通道权重（不破坏预训练）=====
        with torch.no_grad():
            # 前 3 通道：原 RGB
            base.conv1.weight[:, :3] = old_conv.weight
            # 后 3 通道：ΔRGB，初始化为 0（等价“慢慢学 motion”）
            base.conv1.weight[:, 3:] = 0.0

        base.fc = nn.Identity()
        self.backbone = base   # 输出 512

        self.proj = nn.Linear(512, img_dim)

        # -------- Temporal Conv（保持你原来的）--------
        self.temporal_conv = nn.Conv1d(
            in_channels=img_dim,
            out_channels=img_dim,
            kernel_size=ksize,
            padding=ksize // 2
        )

        self.norm = nn.BatchNorm1d(img_dim)
        self.act  = nn.GELU()

    def forward(self, x):
        """
        x: [B,L,3,64,64]
        """
        B, L = x.shape[:2]

        # ===== 帧差 =====
        # ΔI_t = I_t - I_{t-1}，第一帧用 0
        delta = x[:, 1:] - x[:, :-1]                  # [B,L-1,3,64,64]
        zero  = torch.zeros_like(delta[:, :1])
        delta = torch.cat([zero, delta], dim=1)       # [B,L,3,64,64]

        # ===== 拼接 RGB + ΔRGB =====
        x = torch.cat([x, delta], dim=2)               # [B,L,6,64,64]

        # ---- 逐帧 CNN ----
        x = x.reshape(B * L, 6, 64, 64)                # [B*L,6,64,64]
        feat = self.backbone(x)                        # [B*L,512]
        feat = self.proj(feat)                         # [B*L,img_dim]

        feat = feat.reshape(B, L, -1)                  # [B,L,img_dim]

        # ---- 时间卷积（视频建模）----
        feat = feat.transpose(1, 2)                    # [B,img_dim,L]
        feat = self.temporal_conv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = feat.transpose(1, 2)                    # [B,L,img_dim]

        return feat


class PatchTSTWithCNN(nn.Module):
    """
    image → CNN → concat(pv, time) → PatchTST
    不使用任何日内 encoder / pooling
    """
    def __init__(
        self,
        seq_len,
        ct_dim,
        hist_days=1,
        img_dim=128,
        patchtst_d_model=256,
        patchtst_heads=8,
        patchtst_layers=3,
        patch_len=6,
        dropout=0.1,
        cnn_pretrained=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hist_days = hist_days
        self.img_dim = img_dim

        # CNN：逐时间步
        self.cnn = VideoCNN(
            img_dim=img_dim,
            pretrained=cnn_pretrained,
            ksize=3  # 可以试 3 / 5
        )
        self.cloud_phys = CloudPhysicsExtractor()
        self.phys_dim = 4
        self.phys_gate = PhysicsGuidedGate(
            phys_dim=self.phys_dim,
            img_dim=img_dim,
            hidden_dim=64
        )

        # PatchTST 输入维度
        # hist: [img_feat, y_hist, t_hist]
        enc_in_hist = img_dim + 1 + ct_dim+ self.phys_dim
        # enc_in_hist = img_dim + 1

        # fut : [img_feat, t_fut]
        enc_in_fut  = img_dim+ self.phys_dim+ ct_dim
        # enc_in_fut  = img_dim + ct_dim


        self.patchtst = PatchTST_D1(
            enc_in_hist=enc_in_hist,
            enc_in_future=enc_in_fut,
            hist_len=hist_days * seq_len,
            pred_len=seq_len,
            d_model=patchtst_d_model,
            n_heads=patchtst_heads,
            e_layers=patchtst_layers,
            patch_len=patch_len,
            dropout=dropout
        )

    def forward(self, hist_img, y_hist, t_hist, fut_img, t_fut, mask_hist, mask_fut):
        """
        hist_img: [B,L,3,64,64]
        y_hist  : [B,L,1]
        t_hist  : [B,L,Ct]
        fut_img : [B,L,3,64,64]
        t_fut   : [B,L,Ct]
        mask_*  : [B,L]  (这里不直接用，留给 loss)
        """
        # --- CNN 提取逐时间步特征 ---
        # --- CNN 特征 ---
        # --- CNN 特征 ---
        hist_feat = self.cnn(hist_img)  # [B,Lh,img_dim]
        fut_feat = self.cnn(fut_img)  # [B,Lf,img_dim]

        # --- 云物理特征 ---
        hist_phys = self.cloud_phys(hist_img)  # [B,Lh,4]
        fut_phys = self.cloud_phys(fut_img)  # [B,Lf,4]

        # ===== Physics-guided gating（核心新增）=====
        hist_feat = self.phys_gate(hist_feat, hist_phys)
        fut_feat = self.phys_gate(fut_feat, fut_phys)

        # --- 拼接 ---
        hist_in = torch.cat(
            [hist_feat, hist_phys, y_hist, t_hist],
            dim=-1
        )

        fut_in = torch.cat(
            [fut_feat, fut_phys, t_fut],
            dim=-1
        )

        # --- PatchTST ---
        out = self.patchtst(hist_in, fut_in)  # [B,L,1]
        return out


# =======================
# 4) SKIPP’D 数据读取与 15min 聚合（取“每15min第一个样本”）
# =======================
def load_skippd_window(h5_path, times_path, group, start_date, end_date, freq):
    """
    返回：按天组织的 dict:
      days: [day0, day1, ...] (pd.Timestamp normalize)
      day_to_data[day] = {
          "times": pd.DatetimeIndex (15min bins),
          "pv": np.ndarray [Ld,1],
          "img": np.ndarray [Ld,3,64,64] uint8
      }
    """
    times = pd.to_datetime(np.load(times_path, allow_pickle=True))
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    with h5py.File(h5_path, "r") as f:
        imgs = f[f"{group}/images_log"]
        pv   = f[f"{group}/pv_log"]

        df = pd.DataFrame({"time": times})
        df["day"] = df["time"].dt.normalize()
        df = df[(df["day"] >= start_date) & (df["day"] <= end_date)].copy()
        df = df.reset_index(drop=False).rename(columns={"index": "idx"})  # idx 是原数组索引

        # 按天
        days = sorted(df["day"].unique())
        day_to_data = {}

        for d in days:
            sub = df[df["day"] == d].sort_values("time")
            if len(sub) == 0:
                continue

            # 以 time 为 index 做 15min resample，取 first（与你之前一致：每隔15min取一次，不取平均）
            sub2 = sub.set_index("time")
            # first 需要保留 idx
            idx_first = sub2["idx"].resample(freq).first().dropna().astype(int).values
            if len(idx_first) == 0:
                continue

            t_bins = pd.to_datetime(sub2["idx"].resample(freq).first().dropna().index)
            pv_bins = pv[idx_first].astype(np.float32).reshape(-1, 1)
            img_bins = imgs[idx_first]  # uint8 [Ld,64,64,3]

            # 转成 torch 习惯的 [Ld,3,64,64]
            img_bins = np.transpose(img_bins, (0, 3, 1, 2))

            day_to_data[pd.Timestamp(d)] = {
                "times": t_bins,
                "pv": pv_bins,
                "img": img_bins
            }

    # 以“有数据的天序列”为连续时间轴（你已同意）
    days = [d for d in days if pd.Timestamp(d) in day_to_data]
    days = sorted(days)
    return days, day_to_data


def pad_trunc_day(arr, seq_len, pad_value=0.0, is_image=False):
    """
    arr:
      - image: [L,3,64,64]
      - pv   : [L,1]
      - timef: [L,Ct]
    return: padded/truncated, mask_valid [seq_len]
    """
    L = arr.shape[0]
    if L >= seq_len:
        out = arr[:seq_len]
        mask = np.ones((seq_len,), dtype=bool)
        return out, mask

    # pad
    if is_image:
        out = np.zeros((seq_len, *arr.shape[1:]), dtype=arr.dtype)
        out[:L] = arr
    else:
        out = np.full((seq_len, *arr.shape[1:]), pad_value, dtype=arr.dtype)
        out[:L] = arr

    mask = np.zeros((seq_len,), dtype=bool)
    mask[:L] = True
    return out, mask


# =======================
# 5) 你要求的 create_D1_samples：返回变长语义（但内部 pad 到 SEQ_LEN 以适配 PatchTST）
# =======================
def create_D1_samples(days, day_to_data, scaler_y, scaler_t, hist_days=1, pred_days=1, seq_len=54):
    """
    每个样本：
      历史：hist_days 天 -> (x_hist img, y_hist pv, t_hist timef)
      未来：pred_days 天（当前脚本默认 pred_days=1） -> (x_fut img, t_fut timef)
      label: y_fut pv

    说明：
      - 真实步长不一致，但为了不改 PatchTST，这里对“每一天”先 pad/trunc 到固定 seq_len，
        再把 hist_days 天沿时间维拼接成 hist_days*seq_len 的长历史序列，并返回对应 mask。
      - 未来仍为 1 天（seq_len），以适配你的 D+1 设置。
    """
    assert pred_days == 1, "当前脚本实现的是 D+1（pred_days=1）。如需多天预测我再给你扩展。"

    X_hist_img, Y_hist, T_hist, M_hist = [], [], [], []
    X_fut_img,  T_fut,  Y_fut,  M_fut  = [], [], [], []
    fut_dates = []

    # 以“有数据天序列”为连续轴：使用 hist_days 天历史预测下一天
    for i in range(hist_days - 1, len(days) - pred_days):
        # 未来日（D+1）
        d_fut = pd.Timestamp(days[i + 1])

        # ===== 历史 hist_days 天：days[i-hist_days+1 ... i] =====
        hist_days_list = [pd.Timestamp(x) for x in days[i - hist_days + 1: i + 1]]

        hist_imgs_fix, hist_pv_fix, hist_tf_fix, hist_masks = [], [], [], []

        for d_hist in hist_days_list:
            hist_img = day_to_data[d_hist]["img"]  # [Lh,3,64,64]
            hist_pv  = day_to_data[d_hist]["pv"]   # [Lh,1] (pv_log)
            hist_ts  = pd.Series(day_to_data[d_hist]["times"])
            hist_tf  = build_time_features(hist_ts)  # [Lh,Ct]

            # scaler
            hist_pv_s = scaler_y.transform(hist_pv)  # [Lh,1]
            hist_tf_s = scaler_t.transform(hist_tf)  # [Lh,Ct]

            # pad/trunc 到 seq_len（每一天先对齐长度）
            img_fix, m_fix = pad_trunc_day(hist_img, seq_len, is_image=True)
            pv_fix,  _     = pad_trunc_day(hist_pv_s, seq_len, pad_value=0.0, is_image=False)
            tf_fix,  _     = pad_trunc_day(hist_tf_s, seq_len, pad_value=0.0, is_image=False)

            hist_imgs_fix.append(img_fix.astype(np.uint8))
            hist_pv_fix.append(pv_fix.astype(np.float32))
            hist_tf_fix.append(tf_fix.astype(np.float32))
            hist_masks.append(m_fix.astype(bool))

        # 拼接成一条长历史序列：[(seq_len)*hist_days, ...]
        hist_img_cat = np.concatenate(hist_imgs_fix, axis=0)   # [hist_days*seq_len,3,64,64]
        hist_pv_cat  = np.concatenate(hist_pv_fix,  axis=0)    # [hist_days*seq_len,1]
        hist_tf_cat  = np.concatenate(hist_tf_fix,  axis=0)    # [hist_days*seq_len,Ct]
        hist_m_cat   = np.concatenate(hist_masks,   axis=0)    # [hist_days*seq_len]

        # ===== 未来 1 天 =====
        fut_img = day_to_data[d_fut]["img"]
        fut_pv  = day_to_data[d_fut]["pv"]     # label
        fut_ts  = pd.Series(day_to_data[d_fut]["times"])
        fut_tf  = build_time_features(fut_ts)

        fut_pv_s  = scaler_y.transform(fut_pv)             # [Lf,1]
        fut_tf_s  = scaler_t.transform(fut_tf)             # [Lf,Ct]

        fut_img_fix, mf  = pad_trunc_day(fut_img, seq_len, is_image=True)
        fut_tf_fix, _    = pad_trunc_day(fut_tf_s, seq_len, pad_value=0.0, is_image=False)
        fut_pv_fix, _    = pad_trunc_day(fut_pv_s, seq_len, pad_value=0.0, is_image=False)

        # 保存
        X_hist_img.append(hist_img_cat)
        Y_hist.append(hist_pv_cat)
        T_hist.append(hist_tf_cat)
        M_hist.append(hist_m_cat)

        X_fut_img.append(fut_img_fix.astype(np.uint8))
        T_fut.append(fut_tf_fix.astype(np.float32))
        Y_fut.append(fut_pv_fix.astype(np.float32))
        M_fut.append(mf.astype(bool))

        fut_dates.append(pd.Timestamp(d_fut))

    return (
        np.stack(X_hist_img),  # [N, hist_days*seq_len, 3, 64, 64]
        np.stack(Y_hist),      # [N, hist_days*seq_len, 1]
        np.stack(T_hist),      # [N, hist_days*seq_len, Ct]
        np.stack(M_hist),      # [N, hist_days*seq_len] bool
        np.stack(X_fut_img),   # [N, seq_len, 3, 64, 64]
        np.stack(T_fut),       # [N, seq_len, Ct]
        np.stack(Y_fut),       # [N, seq_len, 1]
        np.stack(M_fut),       # [N, seq_len] bool
        pd.to_datetime(fut_dates)
    )


# =======================
# 6) Dataset + masked loss
# =======================
class SKIPPD_D1_Dataset(Dataset):
    def __init__(self, hist_img, y_hist, t_hist, m_hist, fut_img, t_fut, y_fut, m_fut):
        self.hist_img = torch.from_numpy(hist_img)  # uint8
        self.y_hist   = torch.from_numpy(y_hist)    # float32
        self.t_hist   = torch.from_numpy(t_hist)    # float32
        self.m_hist   = torch.from_numpy(m_hist)    # bool

        self.fut_img  = torch.from_numpy(fut_img)   # uint8
        self.t_fut    = torch.from_numpy(t_fut)     # float32
        self.y_fut    = torch.from_numpy(y_fut)     # float32
        self.m_fut    = torch.from_numpy(m_fut)     # bool

    def __len__(self):
        return self.hist_img.shape[0]

    def __getitem__(self, idx):
        return (
            self.hist_img[idx], self.y_hist[idx], self.t_hist[idx], self.m_hist[idx],
            self.fut_img[idx],  self.t_fut[idx],  self.y_fut[idx],  self.m_fut[idx]
        )


def masked_mse(pred, target, mask):
    """
    pred/target: [B,L,1]
    mask: [B,L] bool True=valid
    """
    mask_f = mask.unsqueeze(-1).float()
    diff2 = (pred - target) ** 2 * mask_f
    denom = mask_f.sum().clamp_min(1.0)
    return diff2.sum() / denom


def masked_rmse(pred, target, mask):
    mask_f = mask.float()
    diff2 = (pred - target) ** 2
    diff2 = diff2.squeeze(-1) if diff2.dim() == 3 else diff2
    diff2 = diff2 * mask_f
    denom = mask_f.sum().clamp_min(1.0)
    return torch.sqrt(diff2.sum() / denom)


# =======================
# 7) 主流程：加载 -> 构造样本 -> 训练/早停 -> 按月评估（保持你原口径）
# =======================
def main():
    # --- 7.1 读取并按天聚合 ---
    days, day_to_data = load_skippd_window(
        h5_path=H5_PATH,
        times_path=TIMES_PATH,
        group=GROUP,
        start_date=START_DATE,
        end_date=END_DATE,
        freq=FREQ
    )
    print(f"选定窗口内有数据的天数: {len(days)}")
    assert len(days) >= 2, "窗口内可用天不足，无法构造 D+1"

    # --- 7.2 为 scaler 准备全体数据（只用窗口内） ---
    # y: pv_log（float），t: time_features（float）
    all_pv = []
    all_tf = []
    for d in days:
        pv = day_to_data[pd.Timestamp(d)]["pv"]  # [L,1]
        ts = pd.Series(day_to_data[pd.Timestamp(d)]["times"])
        tf = build_time_features(ts)             # [L,Ct]
        all_pv.append(pv)
        all_tf.append(tf)

    all_pv = np.concatenate(all_pv, axis=0).astype(np.float32)
    all_tf = np.concatenate(all_tf, axis=0).astype(np.float32)

    scaler_y = MinMaxScaler()
    scaler_t = MinMaxScaler()
    scaler_y.fit(all_pv)  # pv_log
    scaler_t.fit(all_tf)

    Ct = all_tf.shape[1]

    # --- 7.3 构造 D+1 样本（核心：create_D1_samples 语义对齐） ---
    (Xh_img, Yh, Th, Mh, Xf_img, Tf, Yf, Mf, fut_dates) = create_D1_samples(
        days=days,
        day_to_data=day_to_data,
        scaler_y=scaler_y,
        scaler_t=scaler_t,
        hist_days=HIST_DAYS,
        pred_days=PRED_DAYS,
        seq_len=SEQ_LEN
    )

    N_samples = Xh_img.shape[0]
    print(f"D+1 样本数: {N_samples}")
    split_idx = int(N_samples * 0.7)

    # --- 7.4 划分 train/val（保持你原来 7:3） ---
    train_set = SKIPPD_D1_Dataset(
        Xh_img[:split_idx], Yh[:split_idx], Th[:split_idx], Mh[:split_idx],
        Xf_img[:split_idx], Tf[:split_idx], Yf[:split_idx], Mf[:split_idx]
    )
    val_set = SKIPPD_D1_Dataset(
        Xh_img[split_idx:], Yh[split_idx:], Th[split_idx:], Mh[split_idx:],
        Xf_img[split_idx:], Tf[split_idx:], Yf[split_idx:], Mf[split_idx:]
    )
    dates_val = fut_dates[split_idx:]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    # --- 7.5 模型（PatchTST 保留，前加 CNN+IntraTrans+Pool） ---
    model = PatchTSTWithCNN(
        seq_len=SEQ_LEN,
        ct_dim=Ct,
        hist_days=HIST_DAYS,
        img_dim=128,
        patchtst_d_model=256,
        patchtst_heads=8,
        patchtst_layers=3,
        patch_len=PATCH_LEN,
        dropout=0.1,
        cnn_pretrained=True
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- 7.6 验证函数（保留你原按月 ACC 口径；RMSE 用 mask_fut） ---
    def evaluate_on_days(model):
        model.eval()
        preds_list, trues_list, masks_list = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                (hist_img, y_hist, t_hist, m_hist,
                 fut_img,  t_fut,  y_fut,  m_fut) = batch

                # images uint8 -> float [0,1]
                hist_img = hist_img.to(DEVICE).float() / 255.0
                fut_img  = fut_img.to(DEVICE).float() / 255.0

                y_hist = y_hist.to(DEVICE)
                t_hist = t_hist.to(DEVICE)
                m_hist = m_hist.to(DEVICE)

                t_fut = t_fut.to(DEVICE)
                y_fut = y_fut.to(DEVICE)
                m_fut = m_fut.to(DEVICE)

                out = model(hist_img, y_hist, t_hist, fut_img, t_fut, m_hist, m_fut)  # [1,L,1]
                pred = out.squeeze(0).cpu().numpy()      # [L,1]
                true = y_fut.squeeze(0).cpu().numpy()    # [L,1]
                mask = m_fut.squeeze(0).cpu().numpy()    # [L]

                preds_list.append(pred.squeeze(-1))      # [L]
                trues_list.append(true.squeeze(-1))      # [L]
                masks_list.append(mask.astype(bool))     # [L]

        preds_arr = np.stack(preds_list)  # [N_val,L]
        trues_arr = np.stack(trues_list)  # [N_val,L]
        masks_arr = np.stack(masks_list)  # [N_val,L] bool

        # 反归一化（保持你原风格）
        preds_inv = scaler_y.inverse_transform(preds_arr.reshape(-1, 1)).reshape(preds_arr.shape)
        trues_inv = scaler_y.inverse_transform(trues_arr.reshape(-1, 1)).reshape(trues_arr.shape)

        # 按月计算 ACC = 1 - RMSE/Cap（RMSE 只在 mask=True 的位置计算）
        dates_series = pd.Series(dates_val)
        months = dates_series.dt.to_period("M")

        results = {}
        for m in np.unique(months):
            mask_m = (months == m).values  # [N_val]
            if mask_m.sum() == 0:
                continue

            p = preds_inv[mask_m]   # [n,L]
            t = trues_inv[mask_m]
            mk = masks_arr[mask_m]  # [n,L]

            # masked RMSE
            diff2 = (p - t) ** 2
            denom = mk.sum()
            if denom <= 0:
                continue
            rmse = np.sqrt(diff2[mk].mean())
            acc = 1 - rmse / capacity
            results[str(m)] = acc

        mean_acc = float(np.mean(list(results.values()))) if len(results) else -1e9
        return mean_acc

    # --- 7.7 训练 + EarlyStopping（保持你原逻辑） ---
    best_acc = -1e9
    best_state = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            (hist_img, y_hist, t_hist, m_hist,
             fut_img,  t_fut,  y_fut,  m_fut) = batch

            hist_img = hist_img.to(DEVICE).float() / 255.0
            fut_img  = fut_img.to(DEVICE).float() / 255.0

            y_hist = y_hist.to(DEVICE)
            t_hist = t_hist.to(DEVICE)
            m_hist = m_hist.to(DEVICE)

            t_fut = t_fut.to(DEVICE)
            y_fut = y_fut.to(DEVICE)
            m_fut = m_fut.to(DEVICE)

            optimizer.zero_grad()
            out = model(hist_img, y_hist, t_hist, fut_img, t_fut, m_hist, m_fut)  # [B,L,1]
            loss = masked_mse(out, y_fut, m_fut)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        val_mean_acc = evaluate_on_days(model)

        print(f"Epoch {epoch+1}/{EPOCHS}   Loss = {train_loss:.6f}   Val_mean_Acc = {val_mean_acc:.6f}")

        # Early Stopping（保留你原逻辑）
        if val_mean_acc > best_acc:
            best_acc = val_mean_acc
            best_state = deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}, best mean acc = {best_acc:.6f}")
                break

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- 7.8 最终按月输出（保持你原输出口径） ---
    model.eval()
    preds_day, trues_day, masks_day = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            (hist_img, y_hist, t_hist, m_hist,
             fut_img,  t_fut,  y_fut,  m_fut) = batch

            hist_img = hist_img.to(DEVICE).float() / 255.0
            fut_img  = fut_img.to(DEVICE).float() / 255.0

            y_hist = y_hist.to(DEVICE)
            t_hist = t_hist.to(DEVICE)
            m_hist = m_hist.to(DEVICE)

            t_fut = t_fut.to(DEVICE)
            y_fut = y_fut.to(DEVICE)
            m_fut = m_fut.to(DEVICE)

            out = model(hist_img, y_hist, t_hist, fut_img, t_fut, m_hist, m_fut)  # [1,L,1]
            pred = out.squeeze(0).squeeze(-1).cpu().numpy()      # [L]
            true = y_fut.squeeze(0).squeeze(-1).cpu().numpy()    # [L]
            mk   = m_fut.squeeze(0).cpu().numpy().astype(bool)   # [L]

            preds_day.append(pred)
            trues_day.append(true)
            masks_day.append(mk)

    preds_day = np.stack(preds_day)  # [N_val,L]
    trues_day = np.stack(trues_day)
    masks_day = np.stack(masks_day)  # bool

    preds_inv = scaler_y.inverse_transform(preds_day.reshape(-1, 1)).reshape(preds_day.shape)
    trues_inv = scaler_y.inverse_transform(trues_day.reshape(-1, 1)).reshape(trues_day.shape)

    dates_test = pd.Series(dates_val)
    months = dates_test.dt.to_period("M")

    results = {}
    for m in np.unique(months):
        mask_m = (months == m).values
        if mask_m.sum() == 0:
            continue
        p = preds_inv[mask_m]
        t = trues_inv[mask_m]
        mk = masks_day[mask_m]

        if mk.sum() <= 0:
            continue

        rmse = np.sqrt(((p - t) ** 2)[mk].mean())
        acc = 1 - rmse / capacity
        results[str(m)] = acc

    d1_df = pd.DataFrame({
        "month": list(results.keys()),
        "D+1_RMSE/Cap": list(results.values())
    })

    print("\n========== SKIPP'D (变长日内->pad+mask) + CNN+IntraTransformer+AttnPool + PatchTST + EarlyStopping ==========")
    print(d1_df)

    os.makedirs("PRE", exist_ok=True)
    out_xlsx = f"PRE/SKIPPD_PatchTST_D1_varlen_{START_DATE}_{END_DATE}.xlsx"
    d1_df.to_excel(out_xlsx, index=False)

    with open(f"PRE/SKIPPD_PatchTST_D1_varlen_{START_DATE}_{END_DATE}.pkl", "wb") as f:
        pickle.dump({
            "pred": preds_inv,
            "true": trues_inv,
            "mask": masks_day,
            "date": dates_test,
            "month_metrics": results,
            "best_mean_acc": best_acc,
            "cfg": {
                "start": START_DATE,
                "end": END_DATE,
                "freq": FREQ,
                "seq_len": SEQ_LEN,
                "patch_len": PATCH_LEN
            }
        }, f)

    print(f"\n已保存：{out_xlsx}")

if __name__ == "__main__":
    main()
