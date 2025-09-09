# mmwave_ecg_tdsi_batch_processing_no_distance_test.py
# 測試版本：只處理前3個文件
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from scipy import signal
import pywt

# ✅ 導入TDSI（Bland–Altman 版）
from tdsi_metrics_ba import (
    TriDomainParams, bandpass_filter, compute_tdsi,
    welch_psd, coherence_band_means, bland_altman
)

# =========================
# 1) 讀檔 / 前處理（沿用原本設計）
# =========================

def load_ecg_data(ecg_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """載入和前處理ECG數據（最後60秒）"""
    print(f"正在載入ECG數據: {os.path.basename(ecg_path)}")
    try:
        ecg_df = pd.read_csv(ecg_path)
        if 'time' not in ecg_df.columns or 'ecg' not in ecg_df.columns:
            print("錯誤: ECG CSV檔案中缺少 'time' 或 'ecg' 欄位。")
            return None, None, None

        df_cleaned = ecg_df.dropna(subset=['ecg'])
        ecg_signal = df_cleaned['ecg'].values
        timestamps_ns = df_cleaned['time'].values

        # ECG固定取樣率130 Hz
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
        print(f"  ECG: {len(ecg_signal)} samples, {len(ecg_signal)/ecg_fs:.1f}s at {ecg_fs} Hz")
        return ecg_signal, ecg_time, ecg_fs
    except Exception as e:
        print(f"載入ECG數據錯誤: {e}")
        return None, None, None

def load_and_preprocess_mmwave(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """載入和前處理mmWave數據（最後60秒）"""
    print(f"正在讀取mmWave檔案: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    fs = 11.11  # 固定取樣率
    df_sorted = df.sort_values(by='Frame_Number')
    waveform = df_sorted['Heart_Waveform'].values
    timestamps = df_sorted['Timestamp'].values
    heart_rates = df_sorted['Heart_Rate'].values
    frame_numbers = df_sorted['Frame_Number'].values

    # 只取最後60秒
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
# 2) 濾波（新版：6種濾波方法）
# =========================

def apply_bandpass(x: np.ndarray, fs: float, low=0.8, high=2.0, order=4) -> np.ndarray:
    """帶通濾波 (0.8-2.0 Hz)"""
    nyq = fs / 2.0
    low_norm = low / nyq
    high_norm = high / nyq
    # 確保頻率在有效範圍內
    low_norm = max(0.001, min(low_norm, 0.999))
    high_norm = max(0.001, min(high_norm, 0.999))
    if low_norm >= high_norm:
        high_norm = low_norm + 0.1
    b, a = signal.butter(order, [low_norm, high_norm], btype='bandpass')
    return signal.filtfilt(b, a, x)

def apply_wavelet_denoise(x: np.ndarray, wavelet='sym4', level=6) -> np.ndarray:
    """小波去噪"""
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

def apply_nlms_filter(x: np.ndarray, reference: Optional[np.ndarray] = None, 
                     filter_length: int = 32, mu: float = 0.01) -> np.ndarray:
    """NLMS自適應濾波器"""
    N = len(x)
    if reference is None:
        # 如果沒有參考訊號，使用延遲版本作為參考
        delay = min(10, N//10)
        reference = np.concatenate([np.zeros(delay), x[:-delay]])
    
    # 初始化
    M = min(filter_length, N//4)  # 濾波器長度
    w = np.zeros(M)  # 濾波器係數
    y = np.zeros(N)  # 輸出訊號
    
    # NLMS演算法
    for n in range(M, N):
        # 輸入向量
        x_n = x[n-M+1:n+1][::-1]  # 反向排列
        
        # 預測輸出
        y[n] = np.dot(w, x_n)
        
        # 誤差
        e = reference[n] - y[n]
        
        # 正規化因子
        norm_factor = np.dot(x_n, x_n) + 1e-8
        
        # 更新權重
        w += (mu * e / norm_factor) * x_n
    
    return y

def process_6_methods(mm: np.ndarray, fs: float, ecg_reference: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    6種濾波方法：
    1. 原始波形
    2. 帶通 (0.8-2.0 Hz)
    3. 小波去噪
    4. NLMS
    5. 帶通 + 小波去噪
    6. 帶通 + 小波去噪 + NLMS
    """
    res = {}
    
    # 1. 原始波形
    res['1_raw'] = mm
    
    # 2. 帶通濾波
    bandpass_filtered = apply_bandpass(mm, fs, low=0.8, high=2.0)
    res['2_bandpass'] = bandpass_filtered
    
    # 3. 小波去噪
    wavelet_filtered = apply_wavelet_denoise(mm)
    res['3_wavelet'] = wavelet_filtered
    
    # 4. NLMS自適應濾波
    nlms_filtered = apply_nlms_filter(mm, reference=ecg_reference)
    res['4_nlms'] = nlms_filtered
    
    # 5. 帶通 + 小波去噪
    bandpass_wavelet = apply_wavelet_denoise(bandpass_filtered)
    res['5_bandpass_wavelet'] = bandpass_wavelet
    
    # 6. 帶通 + 小波去噪 + NLMS
    bandpass_wavelet_nlms = apply_nlms_filter(bandpass_wavelet, reference=ecg_reference)
    res['6_bandpass_wavelet_nlms'] = bandpass_wavelet_nlms
    
    return res

# =========================
# 3) 評估與輸出（TDSI 六指標）
# =========================

def resample_to_mmwave_grid(ecg_sig: np.ndarray, ecg_time: np.ndarray, mm_len: int, fs_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    t_mm = np.arange(mm_len) / fs_mm
    ecg_rs = np.interp(t_mm, ecg_time, ecg_sig)
    return ecg_rs, t_mm

def zstats(x: np.ndarray) -> Tuple[float, float]:
    z = (x - np.mean(x)) / (np.std(x) if np.std(x)>0 else 1.0)
    return float(np.mean(z)), float(np.std(z))

def analyze_methods_with_tdsi(
    methods: Dict[str, np.ndarray],
    ecg_sig: np.ndarray, ecg_time: np.ndarray, fs_mm: float,
    params: Optional[TriDomainParams] = None
) -> pd.DataFrame:
    """分析6種濾波方法的TDSI指標"""

    rows = []
    for key, mm in methods.items():
        # resample ECG to mmWave time grid
        ecg_rs, t_mm = resample_to_mmwave_grid(ecg_sig, ecg_time, len(mm), fs_mm)

        # 計算TDSI指標
        metrics = compute_tdsi(mm, ecg_rs, fs_mm, params=params)

        # Z-score 統計
        zmean_mm, zstd_mm = zstats(mm)
        zmean_ecg, zstd_ecg = zstats(ecg_rs)

        # 方法名稱
        title_stub = {
            '1_raw': 'Raw',
            '2_bandpass': 'Bandpass',
            '3_wavelet': 'Wavelet',
            '4_nlms': 'NLMS',
            '5_bandpass_wavelet': 'Bandpass+Wavelet',
            '6_bandpass_wavelet_nlms': 'Bandpass+Wavelet+NLMS'
        }.get(key, key)

        # 表格行（六指標 + 其他便利指標）
        rows.append({
            "method": title_stub,
            # 六種核心指標：
            "time_S_BA": metrics["S_BA"],
            "time_nDTW": metrics["nDTW"],
            "freq_MSC_mean": metrics["msc_mean"],
            "freq_Delta_f_peak_Hz": metrics["delta_f_peak_Hz"],
            "nl_XSampEn": metrics["XSampEn"],
            "nl_PLV": metrics["PLV"],
            # 其他可引用分數
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
    # 排序：原始→5種處理
    order = [
        "Raw", "Bandpass", "Wavelet", "NLMS",
        "Bandpass+Wavelet", "Bandpass+Wavelet+NLMS"
    ]
    df["order_idx"] = df["method"].apply(lambda m: order.index(m) if m in order else 999)
    df = df.sort_values("order_idx").drop(columns=["order_idx"]).reset_index(drop=True)
    return df

# =========================
# 4) main - 測試版本
# =========================

def main():
    """主程式入口 - 測試版本，只處理前3個文件"""
    print("🚀 開始No_distance資料夾6種濾波測試（前3個文件）...")
    
    # 資料夾路徑
    mmwave_folder = r"../data/mmWave Data/No_distance"
    ecg_folder = r"../data/ECG Data/No_distance"
    
    print(f"\nmmWave資料夾: {mmwave_folder}")
    print(f"ECG資料夾: {ecg_folder}")
    
    if not os.path.exists(mmwave_folder) or not os.path.exists(ecg_folder):
        print(f"❌ 資料夾不存在")
        return
    
    # 獲取前3個文件
    mmwave_files = sorted([f for f in os.listdir(mmwave_folder) if f.endswith('.csv')])[:3]
    ecg_files = sorted([f for f in os.listdir(ecg_folder) if f.endswith('.csv')])[:3]
    
    print(f"測試文件: {len(mmwave_files)} 個mmWave文件和 {len(ecg_files)} 個ECG文件")
    
    # TDSI 參數
    params = TriDomainParams(
        bp_low=0.8, bp_high=2.0, bp_order=4,  # 更新帶通頻率範圍
        dtw_radius_s=None, f0_min=0.8, f0_max=2.0, coh_bw=0.30,
        xsampen_m=2, xsampen_r_ratio=0.2, plv_bw=0.30,
        k_freq_penalty=10.0, ba_alpha=1.0, ba_beta=1.0, ba_use_zscore=True
    )
    
    # 處理每組文件
    all_results = []
    
    for i, (mmwave_file, ecg_file) in enumerate(zip(mmwave_files, ecg_files)):
        print(f"\n{'='*80}")
        print(f"處理第 {i+1}/3 組文件")
        print(f"mmWave: {mmwave_file}")
        print(f"ECG: {ecg_file}")
        print(f"{'='*80}")
        
        mmwave_path = os.path.join(mmwave_folder, mmwave_file)
        ecg_path = os.path.join(ecg_folder, ecg_file)
        
        try:
            # 載入mmWave數據
            mm, ts, hr, frames, fs = load_and_preprocess_mmwave(mmwave_path)
            
            # 載入ECG數據
            ecg_sig, ecg_time, ecg_fs = load_ecg_data(ecg_path)
            
            if ecg_sig is None or ecg_time is None:
                print(f"⚠️ ECG數據載入失敗，跳過此文件組")
                continue
            
            # 重採樣ECG作為NLMS參考訊號
            ecg_reference, _ = resample_to_mmwave_grid(ecg_sig, ecg_time, len(mm), fs)
            
            # 進行濾波處理
            print("🔄 執行6種濾波...")
            results = process_6_methods(mm, fs, ecg_reference=ecg_reference)
            
            # 計算TDSI指標
            print("📊 計算TDSI指標...")
            df = analyze_methods_with_tdsi(results, ecg_sig, ecg_time, fs_mm=fs, params=params)
            
            # 添加文件信息
            df['file_index'] = i + 1
            df['mmwave_file'] = mmwave_file
            df['ecg_file'] = ecg_file
            
            all_results.append(df)
            print(f"✅ 第 {i+1} 組文件處理完成")
            
            # 顯示這個文件的TDSI結果
            print(f"\n文件 {i+1} 的TDSI結果:")
            show_cols = ["method", "TDSI", "time_S_BA", "freq_MSC_mean", "nl_PLV"]
            print(df[show_cols].to_string(index=False))
            
        except Exception as e:
            print(f"❌ 處理第 {i+1} 組文件時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    # 統計結果
    if all_results:
        print(f"\n{'='*100}")
        print(f"測試結果統計")
        print(f"{'='*100}")
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # 建立輸出目錄
        output_dir = "../outputs/No_distance_6_filters_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # 儲存詳細結果
        combined_csv = os.path.join(output_dir, "test_detailed_results.csv")
        combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 測試結果已儲存：{combined_csv}")
        
        # 顯示平均結果
        show_cols = ["method", "time_S_BA", "time_nDTW", "freq_MSC_mean", "nl_XSampEn", "nl_PLV", "TDSI"]
        final_summary = combined_df.groupby('method')[show_cols[1:]].mean().reset_index()
        final_summary.columns = show_cols
        print("\n=== 測試結果：六種核心指標平均值（含 TDSI 總分） ===")
        print(final_summary.to_string(index=False))
        
        # 找出最佳方法
        best_tdsi = final_summary.loc[final_summary['TDSI'].idxmax()]
        print(f"\n🏆 測試中TDSI最高的方法: {best_tdsi['method']} (TDSI={best_tdsi['TDSI']:.4f})")
        
        print(f"\n🎉 測試完成! 成功處理了 {len(all_results)} 個文件組")
        print(f"📁 結果儲存在: {output_dir}")
    else:
        print("❌ 沒有成功處理任何文件")

if __name__ == "__main__":
    main()
