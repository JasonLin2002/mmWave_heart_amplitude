# mmwave_ecg_tdsi_pipeline.py
# 需求：numpy, scipy, matplotlib, pandas, pywt
# 本腳本改用 tdsi_metrics_ba.py 輸出 6 指標表格 + 各種必要圖表
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from scipy import signal
import pywt

# ✅ 這是新的重點：導入您已有的 TDSI（Bland–Altman 版）
from tdsi_metrics_ba import (
    TriDomainParams, bandpass_filter, compute_tdsi,
    welch_psd, coherence_band_means, bland_altman
)

# =========================
# 1) 讀檔 / 前處理（沿用您原本的設計）
# =========================

def load_ecg_data(ecg_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    print(f"Loading ECG data from: {ecg_path}")
    try:
        ecg_df = pd.read_csv(ecg_path)
        if 'time' not in ecg_df.columns or 'ecg' not in ecg_df.columns:
            print("錯誤: ECG CSV 檔案中缺少 'time' 或 'ecg' 欄位。")
            return None, None, None

        df_cleaned = ecg_df.dropna(subset=['ecg'])
        ecg_signal = df_cleaned['ecg'].values
        timestamps_ns = df_cleaned['time'].values

        # 您原先固定 ECG fs=130 Hz
        ecg_fs = 130.0
        times_s = timestamps_ns * 1e-9
        N = len(ecg_signal)
        display_duration = 60.0
        if N < 2:
            return None, None, None

        total_duration = times_s[-1] - times_s[0]
        if total_duration <= display_duration:
            start_idx = 0
        else:
            start_time = times_s[-1] - display_duration
            start_idx = np.searchsorted(times_s, start_time, side="left")

        ecg_signal = ecg_signal[start_idx:]
        times_s = times_s[start_idx:]
        ecg_time = times_s - times_s[0]
        print(f"ECG: {len(ecg_signal)} samples, {len(ecg_signal)/ecg_fs:.1f}s at {ecg_fs} Hz")
        return ecg_signal, ecg_time, ecg_fs
    except Exception as e:
        print(f"Error loading ECG data: {e}")
        return None, None, None


def load_and_preprocess_mmwave(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    print(f"Reading mmWave file: {file_path}")
    df = pd.read_csv(file_path)
    fs = 11.11  # 固定取樣率
    df_sorted = df.sort_values(by='Frame_Number')
    waveform = df_sorted['Heart_Waveform'].values
    timestamps = df_sorted['Timestamp'].values
    heart_rates = df_sorted['Heart_Rate'].values
    frame_numbers = df_sorted['Frame_Number'].values

    # 只取最後 60 秒
    N = len(waveform)
    time_axis = np.arange(N) / fs
    display_duration = 60.0
    if N > 1 and time_axis[-1] > display_duration:
        start_time = time_axis[-1] - display_duration
        start_idx = np.searchsorted(time_axis, start_time, side="left")
    else:
        start_idx = 0

    waveform = waveform[start_idx:]
    timestamps = timestamps[start_idx:]
    heart_rates = heart_rates[start_idx:]
    frame_numbers = frame_numbers[start_idx:]
    return waveform, timestamps, heart_rates, frame_numbers, fs


# =========================
# 2) 濾波（沿用您原本 1+7 組）
# =========================

def apply_detrend(x: np.ndarray) -> np.ndarray:
    return signal.detrend(x)

def apply_bandpass(x: np.ndarray, fs: float, low=0.8, high=2.0, order=4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = signal.butter(order, [low/nyq, high/nyq], btype='bandpass')
    return signal.filtfilt(b, a, x)

def apply_wavelet_denoise(x: np.ndarray, wavelet='sym4', level=6) -> np.ndarray:
    orig_len = len(x)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(x)))
    new_coeffs = list(coeffs)
    # 分段閾值策略（保留心跳相關頻段）
    for i in range(1, len(coeffs)):
        if i == 1:      factor = 1.5
        elif i == 2:    factor = 1.2
        elif i == 3:    factor = 0.8
        elif i == 4:    factor = 0.6
        elif i == 5:    factor = 0.5
        else:           factor = 0.4
        new_coeffs[i] = pywt.threshold(coeffs[i], thr*factor, mode='soft')
    rec = pywt.waverec(new_coeffs, wavelet)[:orig_len]
    return rec

def apply_matched_filter(x: np.ndarray, fs: float, bpm=70) -> np.ndarray:
    f0 = bpm/60.0
    T = 1.0/f0
    n = max(16, int(round(T*fs)))
    t = np.linspace(0, T, n)
    tpl = np.sin(2*np.pi*f0*t) * np.exp(-0.5*((t-T/2)/(T/6))**2)
    tpl = tpl/np.linalg.norm(tpl)
    y = signal.correlate(x, tpl, mode='same')
    if np.max(np.abs(y)) > 0:
        y = y/np.max(np.abs(y)) * (np.max(np.abs(x)) if np.max(np.abs(x))>0 else 1.0)
    return y

def process_8_methods(mm: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
    res = {}
    # 1 原始
    res['1_raw'] = mm
    d = apply_detrend(mm)
    # 2 去趨勢→帶通
    res['2_detrend_bandpass'] = apply_bandpass(d, fs)
    # 3 去趨勢→小波
    res['3_detrend_wavelet'] = apply_wavelet_denoise(d)
    # 4 去趨勢→匹配
    res['4_detrend_matched'] = apply_matched_filter(d, fs)
    # 5 去趨勢→帶通→小波
    bp = res['2_detrend_bandpass']
    res['5_bandpass_wavelet'] = apply_wavelet_denoise(bp)
    # 6 去趨勢→帶通→匹配
    res['6_bandpass_matched'] = apply_matched_filter(bp, fs)
    # 7 去趨勢→帶通→小波→匹配
    res['7_bandpass_wavelet_matched'] = apply_matched_filter(res['5_bandpass_wavelet'], fs)
    # 8 去趨勢→帶通→匹配→小波
    res['8_bandpass_matched_wavelet'] = apply_wavelet_denoise(res['6_bandpass_matched'])
    return res


# =========================
# 3) 評估與輸出（新：TDSI 六指標 + 附加圖）
# =========================

def resample_to_mmwave_grid(ecg_sig: np.ndarray, ecg_time: np.ndarray, mm_len: int, fs_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    t_mm = np.arange(mm_len) / fs_mm
    ecg_rs = np.interp(t_mm, ecg_time, ecg_sig)
    return ecg_rs, t_mm

def zstats(x: np.ndarray) -> Tuple[float, float]:
    z = (x - np.mean(x)) / (np.std(x) if np.std(x)>0 else 1.0)
    return float(np.mean(z)), float(np.std(z))

def plot_overlay_mm_ecg(time: np.ndarray, mm: np.ndarray, ecg: np.ndarray, title: str, outpath: str):
    plt.figure(figsize=(12, 3.2))
    plt.plot(time, mm, label='mmWave', linewidth=0.9)
    plt.plot(time, ecg, label='ECG (resampled)', linewidth=0.8, alpha=0.8)
    plt.xlim(time[0], time[-1]); plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.title(title)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_bland_altman(mm: np.ndarray, ecg: np.ndarray, title: str, outpath: str, use_zscore=True):
    ba = bland_altman(mm, ecg, use_zscore=use_zscore)
    x = (mm + ecg) / 2.0
    d = mm - ecg
    plt.figure(figsize=(5.2, 4.0))
    plt.scatter(x, d, s=6, alpha=0.6)
    plt.axhline(ba['bias'], linestyle='--')
    plt.axhline(ba['loa_lower'], linestyle=':')
    plt.axhline(ba['loa_upper'], linestyle=':')
    plt.title(f"{title}\nBA: bias={ba['bias']:.3f}, sd={ba['sd']:.3f}")
    plt.xlabel('Mean of mmWave & ECG'); plt.ylabel('Difference (mmWave-ECG)')
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_psd(mm: np.ndarray, ecg: np.ndarray, fs: float, title: str, outpath: str):
    f1, P1 = welch_psd(mm, fs)
    f2, P2 = welch_psd(ecg, fs)
    plt.figure(figsize=(5.6, 4.0))
    plt.semilogy(f1, P1, label='mmWave')
    plt.semilogy(f2, P2, label='ECG')
    plt.xlim(0, 4.0); plt.xlabel('Hz'); plt.ylabel('PSD')
    plt.title(title); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_coherence(mm: np.ndarray, ecg: np.ndarray, fs: float, title: str, outpath: str):
    nperseg = min(256, len(mm), len(ecg))
    f, Cxy = signal.coherence(mm, ecg, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    plt.figure(figsize=(5.6, 4.0))
    plt.plot(f, Cxy, linewidth=1.0)
    plt.xlim(0, 4.0); plt.ylim(0, 1.0)
    plt.xlabel('Hz'); plt.ylabel('MSC')
    plt.title(title); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def analyze_methods_with_tdsi(
    methods: Dict[str, np.ndarray],
    ecg_sig: np.ndarray, ecg_time: np.ndarray, fs_mm: float,
    outdir: str, params: Optional[TriDomainParams] = None
) -> pd.DataFrame:

    os.makedirs(outdir, exist_ok=True)
    rows = []
    for key, mm in methods.items():
        # resample ECG to mmWave time grid
        ecg_rs, t_mm = resample_to_mmwave_grid(ecg_sig, ecg_time, len(mm), fs_mm)

        # 預處理：帶通+Z（沿用 compute_tdsi 內部會再處理；這裡圖用原值）
        # 計分
        metrics = compute_tdsi(mm, ecg_rs, fs_mm, params=params)

        # Z-score 統計（輸出到表格）
        zmean_mm, zstd_mm = zstats(mm)
        zmean_ecg, zstd_ecg = zstats(ecg_rs)

        # ---- 輸出圖 ----
        title_stub = {
            '1_raw': 'Raw',
            '2_detrend_bandpass': 'Detrend+Bandpass',
            '3_detrend_wavelet': 'Detrend+Wavelet',
            '4_detrend_matched': 'Detrend+Matched',
            '5_bandpass_wavelet': 'Bandpass+Wavelet',
            '6_bandpass_matched': 'Bandpass+Matched',
            '7_bandpass_wavelet_matched': 'Bandpass+Wavelet+Matched',
            '8_bandpass_matched_wavelet': 'Bandpass+Matched+Wavelet'
        }.get(key, key)

        plot_overlay_mm_ecg(t_mm, mm, ecg_rs, f"{title_stub} vs ECG (z-score shown in table)", os.path.join(outdir, f"{key}_overlay.png"))
        plot_bland_altman(mm, ecg_rs, f"{title_stub} Bland–Altman", os.path.join(outdir, f"{key}_bland_altman.png"))
        plot_psd(mm, ecg_rs, fs_mm, f"{title_stub} PSD (Welch)", os.path.join(outdir, f"{key}_psd.png"))
        plot_coherence(mm, ecg_rs, fs_mm, f"{title_stub} Coherence", os.path.join(outdir, f"{key}_coherence.png"))

        # ---- 表格行（六指標 + 其他便利指標）----
        rows.append({
            "method": title_stub,
            # 六種核心指標：
            "time_S_BA": metrics["S_BA"],
            "time_nDTW": metrics["nDTW"],
            "freq_MSC_mean": metrics["msc_mean"],
            "freq_Delta_f_peak_Hz": metrics["delta_f_peak_Hz"],
            "nl_XSampEn": metrics["XSampEn"],
            "nl_PLV": metrics["PLV"],
            # 其他可引用分數（投稿好用）
            "BA_bias": metrics["BA_bias"],
            "BA_sd": metrics["BA_sd"],
            "BA_loa_width": metrics["BA_loa_width"],
            "S_time": metrics["S_time"],
            "S_freq": metrics["S_freq"],
            "S_nl": metrics["S_nl"],
            "TDSI": metrics["TDSI"],
            # Z-score 統計
            "mm_z_mean": zmean_mm, "mm_z_std": zstd_mm,
            "ecg_z_mean": zmean_ecg, "ecg_z_std": zstd_ecg,
            # 參考主頻
            "f0_Hz": metrics["f0_Hz"]
        })

    df = pd.DataFrame(rows)
    # 排序：原始→七種處理
    order = [
        "Raw","Detrend+Bandpass","Detrend+Wavelet","Detrend+Matched",
        "Bandpass+Wavelet","Bandpass+Matched","Bandpass+Wavelet+Matched","Bandpass+Matched+Wavelet"
    ]
    df["order_idx"] = df["method"].apply(lambda m: order.index(m) if m in order else 999)
    df = df.sort_values("order_idx").drop(columns=["order_idx"]).reset_index(drop=True)
    # CSV
    csv_path = os.path.join(outdir, "tdsi_metrics_summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 指標表格已輸出：{csv_path}")
    return df


# =========================
# 4) 原始+七種濾波對照圖（沿用／微調）
# =========================

def visualize_methods_grid(results: Dict[str, np.ndarray], fs: float,
                           ecg_signal: Optional[np.ndarray] = None,
                           ecg_time: Optional[np.ndarray] = None,
                           outpng: str = "methods_vs_ecg_grid.png"):
    time = np.arange(len(results['1_raw'])) / fs
    methods = [
        ('1_raw', 'Raw'),
        ('2_detrend_bandpass', 'Detrend+Bandpass'),
        ('3_detrend_wavelet', 'Detrend+Wavelet'),
        ('4_detrend_matched', 'Detrend+Matched'),
        ('5_bandpass_wavelet', 'Bandpass+Wavelet'),
        ('6_bandpass_matched', 'Bandpass+Matched'),
        ('7_bandpass_wavelet_matched', 'Bandpass+Wavelet+Matched'),
        ('8_bandpass_matched_wavelet', 'Bandpass+Matched+Wavelet')
    ]
    plt.figure(figsize=(18, 20))
    for i, (k, name) in enumerate(methods, 1):
        ax = plt.subplot(4, 2, i)
        ax.plot(time, results[k], linewidth=0.9, label=f"mmWave {name}")
        ax.set_title(f"{name} vs ECG"); ax.grid(alpha=0.3); ax.set_xlim(0, 60)
        if ecg_signal is not None and ecg_time is not None:
            ecg_rs = np.interp(time, ecg_time, ecg_signal)
            ax2 = ax.twinx()
            ax2.plot(time, ecg_rs, linewidth=0.8, alpha=0.7, label="ECG")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1+lines2, labels1+labels2, loc='upper right', fontsize=8)
        if i > 6: ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
    plt.tight_layout(); plt.savefig(outpng, dpi=220, bbox_inches='tight'); plt.close()
    print(f"📊 對照圖已儲存：{outpng}")


# =========================
# 5) main
# =========================

def main():
    # TODO: 改成您的實際路徑
    mmwave_file_path = r"C:\Users\jk121\Documents\Code\NEW_mmWave_PAPER\Output\merged_csv\45cm\05_20_2025_03_37_00.csv"
    ecg_file_path = r"C:\Users\jk121\Documents\Code\NEW_mmWave_PAPER\Output\ECG Data\CSV\45cm\2025-5-20, 337 AM-1.csv"

    if not os.path.exists(mmwave_file_path):
        print(f"❌ mmWave 檔案不存在: {mmwave_file_path}"); return
    if not os.path.exists(ecg_file_path):
        print(f"❌ ECG 檔案不存在: {ecg_file_path}"); return

    # 載入
    mm, ts, hr, frames, fs = load_and_preprocess_mmwave(mmwave_file_path)
    ecg_sig, ecg_time, ecg_fs = load_ecg_data(ecg_file_path)
    if ecg_sig is None:
        print("❌ ECG 讀取失敗"); return

    # 產生原始+7種濾波
    results = process_8_methods(mm, fs)
    os.makedirs("outputs", exist_ok=True)
    visualize_methods_grid(results, fs, ecg_sig, ecg_time, outpng=os.path.join("outputs", "methods_vs_ecg_grid.png"))

    # TDSI 參數（如需可調整）
    params = TriDomainParams(
        bp_low=0.7, bp_high=3.0, bp_order=4,
        dtw_radius_s=None, f0_min=0.8, f0_max=2.0, coh_bw=0.30,
        xsampen_m=2, xsampen_r_ratio=0.2, plv_bw=0.30,
        k_freq_penalty=10.0, ba_alpha=1.0, ba_beta=1.0, ba_use_zscore=True
    )

    # 計分 + 圖表
    df = analyze_methods_with_tdsi(
        results, ecg_sig, ecg_time, fs_mm=fs,
        outdir=os.path.join("outputs", "per_method_figs"),
        params=params
    )

    # 顯示六指標的簡約表（終端機）
    show_cols = [
        "method", "time_S_BA", "time_nDTW",
        "freq_MSC_mean", "freq_Delta_f_peak_Hz",
        "nl_XSampEn", "nl_PLV", "TDSI"
    ]
    print("\n=== 六種核心指標（含 TDSI 總分） ===")
    print(df[show_cols].to_string(index=False))

if __name__ == "__main__":
    main()
