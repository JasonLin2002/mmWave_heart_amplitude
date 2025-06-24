# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from scipy import linalg
import os
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw
from scipy.stats import spearmanr

# %% [markdown]
# ## 1. Read CSV File and Preprocess Data

# %%
def load_ecg_data(ecg_path):
    """Load and preprocess ECG data (last 60 seconds)."""
    print(f"Loading ECG data from: {ecg_path}")
    
    try:
        ecg_df = pd.read_csv(ecg_path)
        print(f"ECG data shape: {ecg_df.shape}")
        print("\nECG data columns:")
        print(ecg_df.columns.tolist())
        
        # 檢查必要的欄位是否存在
        if 'time' not in ecg_df.columns or 'ecg' not in ecg_df.columns:
            print("錯誤: ECG CSV 檔案中缺少 'time' 或 'ecg' 欄位。")
            return None, None, None
            
        # 移除 'ecg' 欄位中 NaN 的行，並確保 'time' 也相應移除
        df_cleaned = ecg_df.dropna(subset=['ecg'])
        
        ecg_signal = df_cleaned['ecg'].values
        timestamps_ns = df_cleaned['time'].values  # 時間戳（奈秒）
        
        if len(timestamps_ns) < 2:
            print("錯誤: ECG 數據不足以計算採樣率。")
            return None, None, None
        
        # ECG採樣率為130Hz
        ecg_fs = 130.0
        print(f"ECG 設定採樣率: {ecg_fs:.2f} Hz")
        
        # 轉換時間戳為秒並取最後60秒（參考mmWave程式碼68-85行的作法）
        times_s = timestamps_ns * 1e-9
        N = len(ecg_signal)
        total_duration = times_s[-1] - times_s[0] if N > 1 else 0
        display_duration = 60.0
        
        if total_duration <= display_duration:
            start_idx = 0
            end_idx = N
        else:
            end_time = times_s[-1]
            start_time = end_time - display_duration
            start_idx = np.searchsorted(times_s, start_time, side="left")
            end_idx = N
            
        # 取最後60秒的數據
        ecg_signal = ecg_signal[start_idx:end_idx]
        timestamps_ns = timestamps_ns[start_idx:end_idx]
        times_s = times_s[start_idx:end_idx]
        
        # 創建相對時間軸（從0開始）
        ecg_time = times_s - times_s[0]
        
        print(f"ECG signal length: {len(ecg_signal)} samples ({len(ecg_signal)/ecg_fs:.1f} seconds)")
        print(f"ECG time range: {ecg_time[0]:.1f} to {ecg_time[-1]:.1f} seconds")
        
        return ecg_signal, ecg_time, ecg_fs
        
    except Exception as e:
        print(f"Error loading ECG data: {e}")
        return None, None, None

def load_and_preprocess_data(file_path):
    """Read and preprocess radar heart amplitude data (only last 60 seconds)."""
    print(f"Reading file: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    print("\nData preview:")
    print(df.head())
    fs = 1 / 0.090  # Fixed sampling rate
    print(f"\nEstimated sampling rate: {fs:.2f} Hz")
    df_sorted = df.sort_values(by='Frame_Number')
    waveform_data = df_sorted['Heart_Waveform'].values
    timestamps = df_sorted['Timestamp'].values
    heart_rates = df_sorted['Heart_Rate'].values
    frame_numbers = df_sorted['Frame_Number'].values

    # Only keep last 60 seconds
    N = len(waveform_data)
    time_axis_full = np.arange(N) / fs
    total_duration = time_axis_full[-1] if N > 1 else 0
    display_duration = 60.0
    if total_duration <= display_duration:
        start_idx = 0
        end_idx = N
    else:
        end_time = time_axis_full[-1]
        start_time = end_time - display_duration
        start_idx = np.searchsorted(time_axis_full, start_time, side="left")
        end_idx = N
    waveform_data = waveform_data[start_idx:end_idx]
    timestamps = timestamps[start_idx:end_idx]
    heart_rates = heart_rates[start_idx:end_idx]
    frame_numbers = frame_numbers[start_idx:end_idx]
    return waveform_data, timestamps, heart_rates, frame_numbers, fs

# %% [markdown]
# ## 2. Signal Denoising and Processing Functions

# %%
def process_radar_heartbeat(waveform_data, fs):
    """Radar heart amplitude signal processing and denoising workflow."""
    results = {
        'raw': waveform_data,
        'processed': {}
    }
    
    detrended = signal.detrend(waveform_data)
    results['processed']['detrended'] = detrended
    
    # # 中值濾波 - 註解掉不需要的濾波方法
    # medfilt_data = signal.medfilt(detrended, kernel_size=5)
    # results['processed']['medfilt'] = medfilt_data
    
    try:
        low = 0.8 / (fs/2)
        high = 2.0 / (fs/2)
        if 0 < low < 1 and 0 < high < 1:
            b, a = signal.butter(4, [low, high], btype='bandpass')
            bandpassed = signal.filtfilt(b, a, detrended)  # 直接對detrended進行帶通濾波
            results['processed']['bandpass'] = bandpassed
        else:
            print(f"Bandpass filter design failed: frequencies out of range, low={low}, high={high}")
            results['processed']['bandpass'] = detrended
    except Exception as e:
        print(f"Bandpass filter processing error: {e}")
        results['processed']['bandpass'] = detrended
    
    # # 小波去噪 - 註解掉不需要的濾波方法
    # try:
    #     wavelet_denoised = wavelet_denoise(results['processed']['bandpass'])
    #     results['processed']['wavelet'] = wavelet_denoised
    # except Exception as e:
    #     print(f"Wavelet denoising error: {e}")
    #     results['processed']['wavelet'] = results['processed']['bandpass']
    
    # # SVD去噪 - 註解掉不需要的濾波方法
    # if len(waveform_data) > 50:
    #     try:
    #         frame_size = 20
    #         pad_size = frame_size - (len(medfilt_data) % frame_size) if len(medfilt_data) % frame_size != 0 else 0
    #         padded_data = np.pad(medfilt_data, (0, pad_size), 'constant')
    #         n_frames = len(padded_data) // frame_size
    #         data_matrix = padded_data.reshape(n_frames, frame_size).T
    #         
    #         U, s, Vh = linalg.svd(data_matrix, full_matrices=False)
    #         
    #         k = min(3, len(s))
    #         s_filtered = np.copy(s)
    #         s_filtered[k:] = 0
    #         
    #         S_filtered = np.diag(s_filtered)
    #         filtered_matrix = U @ S_filtered @ Vh
    #         svd_denoised = filtered_matrix.T.flatten()[:len(waveform_data)]
    #         results['processed']['svd'] = svd_denoised
    #     except Exception as e:
    #         print(f"SVD denoising error: {e}")
    #         results['processed']['svd'] = results['processed']['wavelet']
    # else:
    #     results['processed']['svd'] = results['processed']['wavelet']
    
    # # PCA去噪 - 註解掉不需要的濾波方法
    # try:
    #     frame_size = 20
    #     pad_size = frame_size - (len(results['processed']['svd']) % frame_size) 
    #     pad_size = pad_size if pad_size != frame_size else 0
    #     padded_data = np.pad(results['processed']['svd'], (0, pad_size), 'constant')
    #     n_frames = len(padded_data) // frame_size
    #     data_matrix = padded_data.reshape(n_frames, frame_size)
    # 
    #     from sklearn.decomposition import PCA
    #     pca = PCA(n_components=3)
    #     components = pca.fit_transform(data_matrix)
    #     
    #     reconstructed = pca.inverse_transform(components)
    #     pca_denoised = reconstructed.flatten()[:len(waveform_data)]
    #     results['processed']['pca'] = pca_denoised
    #     
    #     results['pca_info'] = {
    #         'explained_variance': pca.explained_variance_ratio_,
    #         'components': pca.components_
    #     }
    # except Exception as e:
    #     print(f"PCA processing error: {e}")
    #     results['processed']['pca'] = results['processed']['svd'].copy()
    
    
    return results

# %% [markdown]
# ## 3. Wavelet Denoising Function

# %%
def wavelet_denoise(signal_data, wavelet='sym4', level=4):
    """Denoise signal using wavelet transform."""
    original_length = len(signal_data)
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))

    new_coeffs = list(coeffs)
    for i in range(1, len(coeffs)):
        new_coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

    reconstructed_signal = pywt.waverec(new_coeffs, wavelet)

    return reconstructed_signal[:original_length]

# %% [markdown]
# ## 4. Metrics Calculation Functions

# %%
def calc_metrics(mmwave_signal, mmwave_time, ecg_signal, ecg_time):
    """Calculate comparison metrics between mmWave and ECG signals."""
    # 重採樣 mmWave 信號到 ECG 的時間軸
    mmwave_resampled = np.interp(ecg_time, mmwave_time, mmwave_signal)
    
    # Spearman 相關係數
    spearman_corr, spearman_p = spearmanr(mmwave_resampled, ecg_signal)
    
    # Cross-correlation
    norm_mmwave = mmwave_resampled - np.mean(mmwave_resampled)
    norm_ecg = ecg_signal - np.mean(ecg_signal)
    
    if np.std(norm_mmwave) > 1e-6 and np.std(norm_ecg) > 1e-6:
        cross_corr = signal.correlate(norm_mmwave / np.std(norm_mmwave), 
                                     norm_ecg / np.std(norm_ecg), mode='full')
        lags = signal.correlation_lags(len(norm_mmwave), len(norm_ecg), mode='full')
        max_cross_corr_value = np.max(cross_corr)
        lag_at_max_corr = lags[np.argmax(cross_corr)]
        time_lag = lag_at_max_corr * (ecg_time[1] - ecg_time[0] if len(ecg_time) > 1 else 0)
    else:
        max_cross_corr_value = None
        lag_at_max_corr = None
        time_lag = None
    
    # DTW
    try:
        dtw_result = dtw(np.ascontiguousarray(mmwave_resampled, dtype=np.double),
                        np.ascontiguousarray(ecg_signal, dtype=np.double),
                        keep_internals=True, step_pattern="asymmetric", 
                        open_end=True, open_begin=True)
        dtw_distance = dtw_result.distance
        dtw_normalized = dtw_result.normalizedDistance
    except Exception as e:
        print(f"DTW calculation error: {e}")
        dtw_distance = None
        dtw_normalized = None
    
    return {
        "DTW距離": dtw_distance,
        "DTW正規化距離": dtw_normalized,
        "Spearman": spearman_corr,
        "Spearman_P值": spearman_p,
        "互相關": max_cross_corr_value,
        "互相關延遲(樣本)": lag_at_max_corr,
        "互相關延遲(秒)": time_lag
    }

# %% [markdown]
# ## 5. Visualization of Results

# %%
def visualize_results(results, timestamps, heart_rates, fs, ecg_signal=None, ecg_time=None):
    """Visualize raw and bandpass filtered heart amplitude signals with ECG overlay (last 60 seconds, English labels)."""
    try:
        plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"Font setting error: {e}. English labels may not display correctly.")

    time = np.arange(len(results['raw'])) / fs

    plt.figure(figsize=(15, 6))
    
    # 子圖1: 原始信號
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(time, results['raw'], 'b-', label='mmWave Raw Signal')
    ax1.set_title('mmWave Heart Signal vs Polar H10 ECG Comparison')
    ax1.set_ylabel('mmWave Amplitude', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    ax1.set_xlim(0, 60)
    
    if ecg_signal is not None and ecg_time is not None:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(ecg_time, ecg_signal, 'k-', alpha=0.7, label='ECG Reference')
        ax1_twin.set_ylabel('ECG Amplitude', color='k')
        ax1_twin.tick_params(axis='y', labelcolor='k')
    
    ax1.set_xticks(list(ax1.get_xticks()) + [60])
    ax1.set_xticklabels([str(int(x)) if x == int(x) else f"{x:.1f}" for x in ax1.get_xticks()])

    # # 註解掉中值濾波後信號
    # ax2 = plt.subplot(3, 1, 2)
    # ax2.plot(time, results['processed']['medfilt'], 'g-', label='mmWave Median Filtered')
    # ax2.set_title('Detrended and Median Filtered Signal vs ECG')
    # ax2.set_ylabel('mmWave Amplitude', color='g')
    # ax2.tick_params(axis='y', labelcolor='g')
    # ax2.grid(True)
    # ax2.set_xlim(0, 60)
    # 
    # if ecg_signal is not None and ecg_time is not None:
    #     ax2_twin = ax2.twinx()
    #     ax2_twin.plot(ecg_time, ecg_signal, 'k-', alpha=0.7, label='ECG Reference')
    #     ax2_twin.set_ylabel('ECG Amplitude', color='k')
    #     ax2_twin.tick_params(axis='y', labelcolor='k')
    # 
    # ax2.set_xticks(list(ax2.get_xticks()) + [60])
    # ax2.set_xticklabels([str(int(x)) if x == int(x) else f"{x:.1f}" for x in ax2.get_xticks()])

    # 子圖2: 帶通濾波後信號
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(time, results['processed']['bandpass'], color='red', label='mmWave Bandpass Filtered')
    ax2.set_title('Bandpass Filtered Signal (0.8-2.0 Hz) vs Polar H10 ECG Comparison')
    ax2.set_ylabel('mmWave Amplitude', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True)
    ax2.set_xlim(0, 60)
    ax2.set_xlabel('Time (seconds)')
    
    if ecg_signal is not None and ecg_time is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(ecg_time, ecg_signal, 'k-', alpha=0.7, label='ECG Reference')
        ax2_twin.set_ylabel('ECG Amplitude', color='k')
        ax2_twin.tick_params(axis='y', labelcolor='k')
    
    ax2.set_xticks(list(ax2.get_xticks()) + [60])
    ax2.set_xticklabels([str(int(x)) if x == int(x) else f"{x:.1f}" for x in ax2.get_xticks()])

    # # 註解掉不需要的子圖4: 小波去噪後信號
    # ax4 = plt.subplot(5, 1, 4)
    # ax4.plot(time, results['processed']['wavelet'], 'm-', label='mmWave Wavelet Denoised')
    # ax4.set_title('Wavelet Denoised Signal vs ECG')
    # ax4.set_ylabel('mmWave Amplitude', color='m')
    # ax4.tick_params(axis='y', labelcolor='m')
    # ax4.grid(True)
    # ax4.set_xlim(0, 60)
    # 
    # if ecg_signal is not None and ecg_time is not None:
    #     ax4_twin = ax4.twinx()
    #     ax4_twin.plot(ecg_time, ecg_signal, 'k-', alpha=0.7, label='ECG Reference')
    #     ax4_twin.set_ylabel('ECG Amplitude', color='k')
    #     ax4_twin.tick_params(axis='y', labelcolor='k')
    # 
    # ax4.set_xticks(list(ax4.get_xticks()) + [60])
    # ax4.set_xticklabels([str(int(x)) if x == int(x) else f"{x:.1f}" for x in ax4.get_xticks()])

    # # 註解掉不需要的子圖5: PCA去噪後信號
    # ax5 = plt.subplot(5, 1, 5)
    # ax5.plot(time, results['processed']['pca'], 'c-', label='mmWave PCA Denoised')
    # ax5.set_title('PCA Reconstructed Signal vs ECG')
    # ax5.set_ylabel('mmWave Amplitude', color='c')
    # ax5.tick_params(axis='y', labelcolor='c')
    # ax5.grid(True)
    # ax5.set_xlim(0, 60)
    # ax5.set_xlabel('Time (seconds)')
    # 
    # if ecg_signal is not None and ecg_time is not None:
    #     ax5_twin = ax5.twinx()
    #     ax5_twin.plot(ecg_time, ecg_signal, 'k-', alpha=0.7, label='ECG Reference')
    #     ax5_twin.set_ylabel('ECG Amplitude', color='k')
    #     ax5_twin.tick_params(axis='y', labelcolor='k')
    # 
    # ax5.set_xticks(list(ax5.get_xticks()) + [60])
    # ax5.set_xticklabels([str(int(x)) if x == int(x) else f"{x:.1f}" for x in ax5.get_xticks()])

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Main Function

# %%
def main():
    """Main program entry."""
    # mmWave數據路徑
    file_path = "/path/to/mmwave/data.csv"
    
    # ECG數據路徑
    ecg_path = "/path/to/ecg/data.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        file_path = input("Please enter the full path to the mmWave CSV file: ")
    
    # 載入mmWave數據
    waveform_data, timestamps, heart_rates, frame_numbers, fs = load_and_preprocess_data(file_path)
    
    # 載入ECG數據
    print("\n" + "="*50)
    ecg_signal, ecg_time, ecg_fs = load_ecg_data(ecg_path)
    print("="*50)
    
    print("\nProcessing heart amplitude data...")
    results = process_radar_heartbeat(waveform_data, fs)
    print("Processing complete!\n")
    
    original_snr = np.mean(results['raw']**2) / np.var(results['raw'])
    processed_snr = np.mean(results['processed']['bandpass']**2) / np.var(results['processed']['bandpass'])
    print(f"SNR improvement (bandpass filter): {processed_snr/original_snr:.2f} times")
    
    # 計算比較指標
    if ecg_signal is not None and ecg_time is not None:
        print("\n" + "="*80)
        print("mmWave信號與ECG比較分析結果")
        print("="*80)
        
        # 標準化信號以進行公平比較
        def normalize_signal(signal):
            return (signal - np.mean(signal)) / np.std(signal)
        
        # 創建mmWave時間軸
        mmwave_time = np.arange(len(results['raw'])) / fs
        
        # 重採樣ECG信號到mmWave的時間軸進行比較
        ecg_resampled = np.interp(mmwave_time, ecg_time, ecg_signal)
        ecg_norm = normalize_signal(ecg_resampled)
        
        # 定義需要比較的方法
        comparison_methods = {
            'raw': '原始信號',
            'bandpass': '帶通濾波'
        }
        
        # 計算各種比較指標
        dtw_distances = {}
        dtw_normalized_distances = {}
        spearman_correlations = {}
        cross_correlations = {}
        
        for method, chinese_name in comparison_methods.items():
            if method == 'raw':
                signal_data = results['raw']
            else:
                signal_data = results['processed'][method]
            
            signal_norm = normalize_signal(signal_data)
            
            # DTW距離計算 (原始DTW距離)
            try:
                dtw_distance = dtw.distance(signal_norm, ecg_norm)
                dtw_distances[method] = dtw_distance
            except Exception as e:
                print(f"DTW計算錯誤 ({chinese_name}): {e}")
                dtw_distances[method] = float('inf')
            
            # DTW正規化距離計算 (額外的DTW距離)
            try:
                # 使用DTW的完整計算結果，計算正規化距離
                dtw_distance_raw = dtw.distance(signal_norm, ecg_norm)
                # 正規化距離 = 原始距離 / 路徑長度
                path_length = max(len(signal_norm), len(ecg_norm))
                dtw_normalized = dtw_distance_raw / path_length
                dtw_normalized_distances[method] = dtw_normalized
            except Exception as e:
                print(f"DTW正規化距離計算錯誤 ({chinese_name}): {e}")
                dtw_normalized_distances[method] = float('inf')
            
            # Spearman相關性計算
            try:
                correlation, p_value = spearmanr(signal_norm, ecg_norm)
                spearman_correlations[method] = correlation
            except Exception as e:
                print(f"Spearman計算錯誤 ({chinese_name}): {e}")
                spearman_correlations[method] = 0.0
            
            # 互相關計算
            try:
                cross_corr_full = np.correlate(signal_norm, ecg_norm, mode='full')
                max_cross_corr = np.max(np.abs(cross_corr_full))
                cross_correlations[method] = max_cross_corr
            except Exception as e:
                print(f"互相關計算錯誤 ({chinese_name}): {e}")
                cross_correlations[method] = 0.0
        
        # 以表格形式顯示結果
        print("\nmmWave信號與ECG比較表:")
        print("-" * 100)
        print("| {:^12} | {:^15} | {:^18} | {:^18} | {:^15} |".format(
            "信號類型", "DTW距離", "DTW正規化距離", "Spearman相關性", "互相關係數"))
        print("|" + "-"*14 + "+" + "-"*17 + "+" + "-"*20 + "+" + "-"*20 + "+" + "-"*17 + "|")
        
        for method, chinese_name in comparison_methods.items():
            dtw_val = dtw_distances.get(method, float('inf'))
            dtw_norm_val = dtw_normalized_distances.get(method, float('inf'))
            spearman_val = spearman_correlations.get(method, 0.0)
            cross_corr_val = cross_correlations.get(method, 0.0)
            
            print("| {:^12} | {:^15.6f} | {:^18.6f} | {:^18.6f} | {:^15.6f} |".format(
                chinese_name, dtw_val, dtw_norm_val, spearman_val, cross_corr_val))
        
        print("-" * 100)
        
        # 總結最佳方法
        print("\n最佳信號處理方法總結:")
        print("-" * 60)
        
        # DTW距離越小越好
        if dtw_distances:
            best_dtw = min(dtw_distances.items(), key=lambda x: x[1])
            print(f"• DTW距離最小 (與ECG最相似): {comparison_methods[best_dtw[0]]:<10} ({best_dtw[1]:.6f})")
        
        # DTW正規化距離越小越好
        if dtw_normalized_distances:
            best_dtw_norm = min(dtw_normalized_distances.items(), key=lambda x: x[1])
            print(f"• DTW正規化距離最小: {comparison_methods[best_dtw_norm[0]]:<10} ({best_dtw_norm[1]:.6f})")
        
        # Spearman相關性越大越好
        if spearman_correlations:
            best_spearman = max(spearman_correlations.items(), key=lambda x: x[1])
            print(f"• Spearman相關性最高: {comparison_methods[best_spearman[0]]:<10} ({best_spearman[1]:.6f})")
        
        # 互相關越大越好
        if cross_correlations:
            best_cross_corr = max(cross_correlations.items(), key=lambda x: x[1])
            print(f"• 互相關係數最高: {comparison_methods[best_cross_corr[0]]:<10} ({best_cross_corr[1]:.6f})")
        
        print("="*80)
    
    print("\nGenerating visualization results with ECG overlay...")
    visualize_results(results, timestamps, heart_rates, fs, ecg_signal, ecg_time)
    
    return results, ecg_signal, ecg_time

if __name__ == "__main__":
    results, ecg_signal, ecg_time = main()
