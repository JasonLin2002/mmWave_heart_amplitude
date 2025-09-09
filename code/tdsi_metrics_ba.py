# tdsi_metrics.py  (Bland–Altman 版本)
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from scipy import signal
from scipy.spatial.distance import cdist

# ========= 基本工具 =========

def bandpass_filter(x: np.ndarray, fs: float,
                    lowcut: float = 0.7, highcut: float = 3.0,
                    order: int = 4, zscore: bool = True, detrend: bool = True) -> np.ndarray:
    x = np.asarray(x, float)
    if detrend:
        x = signal.detrend(x, type="linear")
    nyq = fs / 2.0
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = signal.butter(order, [low, high], btype="bandpass")
    y = signal.filtfilt(b, a, x, method="gust")
    if zscore:
        std = np.std(y)
        if std > 0:
            y = (y - np.mean(y)) / std
        else:
            y = y - np.mean(y)
    return y

def welch_psd(x: np.ndarray, fs: float, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    if nperseg is None:
        nperseg = min(256, len(x))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd

def dominant_peak_freq(x: np.ndarray, fs: float,
                       fmin: float = 0.8, fmax: float = 2.0,
                       nperseg: Optional[int] = None) -> Optional[float]:
    freqs, Pxx = welch_psd(x, fs, nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None
    idx = np.argmax(Pxx[mask])
    return float(freqs[mask][idx])

def normalized_dtw(x: np.ndarray, y: np.ndarray, radius: Optional[int] = None) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    N, M = len(x), len(y)
    cost = np.full((N + 1, M + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0
    for i in range(1, N + 1):
        j_min, j_max = 1, M
        if radius is not None:
            j_min = max(1, i - radius)
            j_max = min(M, i + radius)
        for j in range(j_min, j_max + 1):
            d = abs(x[i - 1] - y[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[N, M] / (N + M))

def avg_in_band(freqs: np.ndarray, values: np.ndarray, f_lo: float, f_hi: float) -> Optional[float]:
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return None
    return float(np.nanmean(values[mask]))

# ========= 頻域／相干 =========

def coherence_band_means(x: np.ndarray, y: np.ndarray, fs: float,
                         f0: float, bw: float = 0.30, nperseg: Optional[int] = None) -> Dict[str, Optional[float]]:
    if nperseg is None:
        nperseg = min(256, len(x), len(y))
    f, Cxy = signal.coherence(x, y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    half = bw / 2.0
    m0 = avg_in_band(f, Cxy, max(0.0, f0 - half), f0 + half)
    m2 = None
    if 2 * f0 < fs / 2 - 1e-6:
        m2 = avg_in_band(f, Cxy, max(0.0, 2 * f0 - half), 2 * f0 + half)
    vals = [v for v in (m0, m2) if v is not None]
    return {
        "msc_f0": m0,
        "msc_2f0": m2,
        "msc_mean": float(np.mean(vals)) if len(vals) > 0 else None,
    }

def peak_freq_error(x: np.ndarray, y: np.ndarray, fs: float,
                    fmin: float = 0.8, fmax: float = 2.0) -> Optional[float]:
    f1 = dominant_peak_freq(x, fs, fmin, fmax)
    f2 = dominant_peak_freq(y, fs, fmin, fmax)
    if f1 is None or f2 is None:
        return None
    return float(abs(f1 - f2))

# ========= 非線性 =========

def xsampen(u: np.ndarray, v: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    u = np.asarray(u, float); v = np.asarray(v, float)
    Nu, Nv = len(u), len(v)
    if Nu <= m + 1 or Nv <= m + 1:
        return np.inf
    std_u, std_v = np.std(u), np.std(v)
    pooled = np.sqrt((std_u**2 + std_v**2) / 2.0)
    r = r_ratio * pooled if pooled > 0 else r_ratio

    def _embed(x: np.ndarray, dim: int) -> np.ndarray:
        return np.array([x[i:i + dim] for i in range(len(x) - dim + 1)])

    U_m = _embed(u, m); V_m = _embed(v, m)
    U_m1 = _embed(u, m + 1); V_m1 = _embed(v, m + 1)

    Dm = cdist(U_m, V_m, metric="chebyshev")
    Dm1 = cdist(U_m1, V_m1, metric="chebyshev")
    B = np.sum(Dm <= r); A = np.sum(Dm1 <= r)
    eps = 1e-12
    return float(-np.log((A + eps) / (B + eps)))

def plv_narrowband(x: np.ndarray, y: np.ndarray, fs: float,
                   f0: float, bw: float = 0.30, order: int = 4) -> float:
    nyq = fs / 2.0
    lo = max(0.05, f0 - bw / 2.0); hi = min(nyq * 0.999, f0 + bw / 2.0)
    if not (0 < lo < hi < nyq):
        return 0.0
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    x_f = signal.filtfilt(b, a, x, method="gust")
    y_f = signal.filtfilt(b, a, y, method="gust")
    phi_x = np.angle(signal.hilbert(x_f))
    phi_y = np.angle(signal.hilbert(y_f))
    plv = np.abs(np.mean(np.exp(1j * (phi_x - phi_y))))
    return float(np.clip(plv, 0.0, 1.0))

# ========= Bland–Altman =========

def bland_altman(x: np.ndarray, y: np.ndarray, use_zscore: bool = True) -> Dict[str, float]:
    """
    Bland–Altman：回傳 bias, sd, loa_lower, loa_upper, loa_width。
    預設先對 x,y 做 Z-score（建議兩者已同單位或已正規化）。
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    L = min(len(x), len(y))
    x = x[:L]; y = y[:L]
    if use_zscore:
        def _zs(a):
            s = np.std(a)
            return (a - np.mean(a)) / s if s > 0 else a - np.mean(a)
        x = _zs(x); y = _zs(y)

    d = x - y
    bias = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    loa_lower = float(bias - 1.96 * sd)
    loa_upper = float(bias + 1.96 * sd)
    loa_width = float(loa_upper - loa_lower)  # = 3.92 * sd
    return {
        "bias": bias,
        "sd": sd,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "loa_width": loa_width,
    }

def bland_altman_score(bias: float, sd: float, alpha: float = 1.0, beta: float = 1.0) -> float:
    """
    將 BA 的 (bias, sd) 映成 0–1 分數；alpha/beta 控制尺度，越小越嚴格。
    S_BA = 0.5*(exp(-|bias|/alpha) + exp(-sd/beta))
    """
    bias_term = np.exp(-abs(bias) / max(alpha, 1e-12))
    sd_term = np.exp(-sd / max(beta, 1e-12))
    return float(0.5 * (bias_term + sd_term))

# ========= 綜合分數（TDSI） =========

@dataclass
class TriDomainParams:
    # 預處理
    bp_low: float = 0.7
    bp_high: float = 3.0
    bp_order: int = 4
    # DTW
    dtw_radius_s: Optional[float] = None
    # 主頻搜尋與頻域
    f0_min: float = 0.8
    f0_max: float = 2.0
    coh_bw: float = 0.30
    welch_nperseg: Optional[int] = None
    # 非線性
    xsampen_m: int = 2
    xsampen_r_ratio: float = 0.2
    plv_bw: float = 0.30
    # TDSI 頻率誤差懲罰
    k_freq_penalty: float = 10.0
    # Bland–Altman 分數尺度
    ba_alpha: float = 1.0
    ba_beta: float = 1.0
    ba_use_zscore: bool = True

def compute_tdsi(mm: np.ndarray, ecg: np.ndarray, fs: float,
                 params: Optional[TriDomainParams] = None) -> Dict[str, float]:
    """
    計算三維度指標（時域改為 Bland–Altman + nDTW）。
    回傳 dict：含 BA 指標、nDTW、頻域、非線性與 TDSI。
    """
    if params is None:
        params = TriDomainParams()

    # 預處理
    mm_f = bandpass_filter(mm, fs, params.bp_low, params.bp_high, params.bp_order, zscore=True, detrend=True)
    ecg_f = bandpass_filter(ecg, fs, params.bp_low, params.bp_high, params.bp_order, zscore=True, detrend=True)

    # ---- 時域：Bland–Altman + nDTW ----
    ba = bland_altman(mm_f, ecg_f, use_zscore=params.ba_use_zscore)
    # nDTW：可選 Sakoe-Chiba 半徑（以秒轉樣本）
    radius = None if params.dtw_radius_s is None else max(1, int(round(params.dtw_radius_s * fs)))
    n_dtw = normalized_dtw(mm_f, ecg_f, radius=radius)
    S_BA = bland_altman_score(ba["bias"], ba["sd"], alpha=params.ba_alpha, beta=params.ba_beta)
    S_time = 0.5 * (S_BA + float(np.exp(-n_dtw)))

    # ---- 頻域 ----
    f0 = dominant_peak_freq(ecg_f, fs, params.f0_min, params.f0_max) or 1.0
    coh = coherence_band_means(mm_f, ecg_f, fs, f0=f0, bw=params.coh_bw, nperseg=params.welch_nperseg)
    msc_mean = coh["msc_mean"]
    df_peak = peak_freq_error(mm_f, ecg_f, fs, params.f0_min, params.f0_max)
    msc_term = 0.0 if msc_mean is None else float(np.clip(msc_mean, 0.0, 1.0))
    freq_penalty = 0.0 if df_peak is None else float(np.exp(-params.k_freq_penalty * abs(df_peak)))
    S_freq = 0.5 * (msc_term + freq_penalty)

    # ---- 非線性 ----
    xse = xsampen(mm_f, ecg_f, m=params.xsampen_m, r_ratio=params.xsampen_r_ratio)
    plv = plv_narrowband(mm_f, ecg_f, fs, f0=f0, bw=params.plv_bw)
    xse_term = float(1.0 / (1.0 + max(xse, 0.0))) if np.isfinite(xse) else 0.0
    S_nl = 0.5 * (xse_term + float(np.clip(plv, 0.0, 1.0)))

    # ---- 總分 ----
    TDSI = float((S_time + S_freq + S_nl) / 3.0)

    return {
        # 時域（Bland–Altman & nDTW）
        "BA_bias": float(ba["bias"]),
        "BA_sd": float(ba["sd"]),
        "BA_loa_lower": float(ba["loa_lower"]),
        "BA_loa_upper": float(ba["loa_upper"]),
        "BA_loa_width": float(ba["loa_width"]),
        "nDTW": float(n_dtw),
        "S_BA": float(S_BA),
        "S_time": float(S_time),
        # 頻域
        "f0_Hz": float(f0),
        "msc_f0": float(coh["msc_f0"]) if coh["msc_f0"] is not None else np.nan,
        "msc_2f0": float(coh["msc_2f0"]) if coh["msc_2f0"] is not None else np.nan,
        "msc_mean": float(msc_mean) if msc_mean is not None else np.nan,
        "delta_f_peak_Hz": float(df_peak) if df_peak is not None else np.nan,
        "S_freq": float(S_freq),
        # 非線性
        "XSampEn": float(xse),
        "PLV": float(plv),
        "S_nl": float(S_nl),
        # 總分
        "TDSI": float(TDSI),
    }

# ========= 便利包裝 =========

def time_domain_metrics(mm: np.ndarray, ecg: np.ndarray, fs: float,
                        dtw_radius_s: Optional[float] = None,
                        ba_alpha: float = 1.0, ba_beta: float = 1.0,
                        ba_use_zscore: bool = True) -> Dict[str, float]:
    """
    時域（Bland–Altman + nDTW）單獨計算。
    """
    mm_f = bandpass_filter(mm, fs)
    ecg_f = bandpass_filter(ecg, fs)
    ba = bland_altman(mm_f, ecg_f, use_zscore=ba_use_zscore)
    radius = None if dtw_radius_s is None else max(1, int(round(dtw_radius_s * fs)))
    ndtw = normalized_dtw(mm_f, ecg_f, radius=radius)
    S_BA = bland_altman_score(ba["bias"], ba["sd"], alpha=ba_alpha, beta=ba_beta)
    S_time = 0.5 * (S_BA + float(np.exp(-ndtw)))
    return {
        "BA_bias": float(ba["bias"]),
        "BA_sd": float(ba["sd"]),
        "BA_loa_lower": float(ba["loa_lower"]),
        "BA_loa_upper": float(ba["loa_upper"]),
        "BA_loa_width": float(ba["loa_width"]),
        "nDTW": float(ndtw),
        "S_BA": float(S_BA),
        "S_time": float(S_time),
    }

def freq_domain_metrics(mm: np.ndarray, ecg: np.ndarray, fs: float,
                        f0_min: float = 0.8, f0_max: float = 2.0,
                        coh_bw: float = 0.30) -> Dict[str, float]:
    mm_f = bandpass_filter(mm, fs)
    ecg_f = bandpass_filter(ecg, fs)
    f0 = dominant_peak_freq(ecg_f, fs, f0_min, f0_max) or 1.0
    coh = coherence_band_means(mm_f, ecg_f, fs, f0=f0, bw=coh_bw)
    df = peak_freq_error(mm_f, ecg_f, fs, f0_min, f0_max)
    return {
        "f0_Hz": float(f0),
        "msc_f0": float(coh["msc_f0"]) if coh["msc_f0"] is not None else np.nan,
        "msc_2f0": float(coh["msc_2f0"]) if coh["msc_2f0"] is not None else np.nan,
        "msc_mean": float(coh["msc_mean"]) if coh["msc_mean"] is not None else np.nan,
        "delta_f_peak_Hz": float(df) if df is not None else np.nan,
    }

def nonlinear_metrics(mm: np.ndarray, ecg: np.ndarray, fs: float,
                      f0_hint: Optional[float] = None, plv_bw: float = 0.30,
                      m: int = 2, r_ratio: float = 0.2) -> Dict[str, float]:
    mm_f = bandpass_filter(mm, fs)
    ecg_f = bandpass_filter(ecg, fs)
    f0 = f0_hint or dominant_peak_freq(ecg_f, fs) or 1.0
    xse = xsampen(mm_f, ecg_f, m=m, r_ratio=r_ratio)
    p = plv_narrowband(mm_f, ecg_f, fs, f0=f0, bw=plv_bw)
    return {"XSampEn": float(xse), "PLV": float(p)}
