# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from scipy import linalg
import os
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from dtaidistance import dtw

# %% [markdown]
# ## 1. Read CSV File and Preprocess Data

# %%
def load_and_preprocess_data(file_path):
    """Read and preprocess radar heart amplitude data (only last 60 seconds)."""
    print(f"Reading file: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    print("\nData preview:")
    print(df.head())
    fs = 11.11  # Fixed sampling rate
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
    
    medfilt_data = signal.medfilt(detrended, kernel_size=5)
    results['processed']['medfilt'] = medfilt_data
    
    try:
        low = 0.8 / (fs/2)
        high = 2.0 / (fs/2)
        if 0 < low < 1 and 0 < high < 1:
            b, a = signal.butter(4, [low, high], btype='bandpass')
            bandpassed = signal.filtfilt(b, a, medfilt_data)
            results['processed']['bandpass'] = bandpassed
        else:
            print(f"Bandpass filter design failed: frequencies out of range, low={low}, high={high}")
            results['processed']['bandpass'] = medfilt_data
    except Exception as e:
        print(f"Bandpass filter processing error: {e}")
        results['processed']['bandpass'] = medfilt_data
    
    try:
        wavelet_denoised = wavelet_denoise(results['processed']['bandpass'])
        results['processed']['wavelet'] = wavelet_denoised
    except Exception as e:
        print(f"Wavelet denoising error: {e}")
        results['processed']['wavelet'] = results['processed']['bandpass']
    
    if len(waveform_data) > 50:
        try:
            frame_size = 20
            pad_size = frame_size - (len(medfilt_data) % frame_size) if len(medfilt_data) % frame_size != 0 else 0
            padded_data = np.pad(medfilt_data, (0, pad_size), 'constant')
            n_frames = len(padded_data) // frame_size
            data_matrix = padded_data.reshape(n_frames, frame_size).T
            
            U, s, Vh = linalg.svd(data_matrix, full_matrices=False)
            
            k = min(3, len(s))
            s_filtered = np.copy(s)
            s_filtered[k:] = 0
            
            S_filtered = np.diag(s_filtered)
            filtered_matrix = U @ S_filtered @ Vh
            svd_denoised = filtered_matrix.T.flatten()[:len(waveform_data)]
            results['processed']['svd'] = svd_denoised
        except Exception as e:
            print(f"SVD denoising error: {e}")
            results['processed']['svd'] = results['processed']['wavelet']
    else:
        results['processed']['svd'] = results['processed']['wavelet']
    
    try:
        frame_size = 20
        pad_size = frame_size - (len(results['processed']['svd']) % frame_size) 
        pad_size = pad_size if pad_size != frame_size else 0
        padded_data = np.pad(results['processed']['svd'], (0, pad_size), 'constant')
        n_frames = len(padded_data) // frame_size
        data_matrix = padded_data.reshape(n_frames, frame_size)
    
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        components = pca.fit_transform(data_matrix)
        
        reconstructed = pca.inverse_transform(components)
        pca_denoised = reconstructed.flatten()[:len(waveform_data)]
        results['processed']['pca'] = pca_denoised
        
        results['pca_info'] = {
            'explained_variance': pca.explained_variance_ratio_,
            'components': pca.components_
        }
    except Exception as e:
        print(f"PCA processing error: {e}")
        results['processed']['pca'] = results['processed']['svd'].copy()
    
    try:
        analytic_signal = signal.hilbert(results['processed']['pca'])
        amplitude_envelope = np.abs(analytic_signal)
        b, a = signal.butter(3, 0.1, btype='lowpass')
        smooth_envelope = signal.filtfilt(b, a, amplitude_envelope)
        results['processed']['envelope'] = smooth_envelope
    except Exception as e:
        print(f"Hilbert transform error: {e}")
        results['processed']['envelope'] = results['processed']['svd']
    
    scaler = StandardScaler()
    results['processed']['final'] = scaler.fit_transform(results['processed']['envelope'].reshape(-1, 1)).flatten()
    
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
# ## 4. Signal Comparison Functions

# %%
def calculate_signal_comparisons(raw_signal, processed_signals):
    #print("\n" + "="*80)
    #print("信號比較分析結果")
    #print("="*80)
    
    # 標準化信號以進行公平比較
    def normalize_signal(signal):
        return (signal - np.mean(signal)) / np.std(signal)
    
    raw_norm = normalize_signal(raw_signal)
    
    # 定義四種濾波方法
    filter_methods = {
        'medfilt': '中值濾波',
        'bandpass': '帶通濾波',
        'wavelet': '小波去噪',
        'pca': 'PCA重建'
    }
    
    # 計算各種比較指標
    dtw_distances = {}
    spearman_correlations = {}
    spearman_pvalues = {}
    cross_correlations = {}
    composite_scores = {}
    
    for method, chinese_name in filter_methods.items():
        if method in processed_signals:
            processed_norm = normalize_signal(processed_signals[method])
            
            # DTW距離計算
            try:
                dtw_distance = dtw.distance(raw_norm, processed_norm)
                dtw_distances[method] = dtw_distance
            except Exception as e:
                print(f"DTW計算錯誤 ({chinese_name}): {e}")
                dtw_distances[method] = float('inf')
            
            # Spearman相關性計算
            try:
                correlation, p_value = spearmanr(raw_norm, processed_norm)
                spearman_correlations[method] = correlation
                spearman_pvalues[method] = p_value
            except Exception as e:
                print(f"Spearman計算錯誤 ({chinese_name}): {e}")
                spearman_correlations[method] = 0.0
                spearman_pvalues[method] = 1.0
            
            # 互相關計算 (與分析最好CVS.py一致的方法)
            try:
                # 使用numpy的correlate函數，取最大值
                cross_corr_full = np.correlate(raw_norm, processed_norm, mode='full')
                max_cross_corr = np.max(np.abs(cross_corr_full))
                cross_correlations[method] = max_cross_corr
            except Exception as e:
                print(f"互相關計算錯誤 ({chinese_name}): {e}")
                cross_correlations[method] = 0.0
            
            # 計算綜合表現分數 (使用dtw_absolute權重方案: DTW 90%, Spearman 5%, Cross-correlation 5%)
            try:
                dtw_score = 1 / (1 + dtw_distances[method]) if dtw_distances[method] != float('inf') else 0
                spearman_score = abs(spearman_correlations[method]) if not np.isnan(spearman_correlations[method]) else 0
                cross_corr_score = abs(cross_correlations[method]) / 1000 if cross_correlations[method] != 0 else 0
                
                # dtw_absolute權重方案: DTW 90%, Spearman 5%, Cross-correlation 5%
                composite_score = (dtw_score * 0.90 + spearman_score * 0.05 + cross_corr_score * 0.05)
                composite_scores[method] = composite_score
            except Exception as e:
                print(f"綜合分數計算錯誤 ({chinese_name}): {e}")
                composite_scores[method] = 0.0
    
    
    comparison_results = {
        'dtw_distances': dtw_distances,
        'spearman_correlations': spearman_correlations,
        'spearman_pvalues': spearman_pvalues,
        'cross_correlations': cross_correlations,
        'composite_scores': composite_scores
    }
    
    return comparison_results

# %% [markdown]
# ## 5. Visualization of Results

# %%
def visualize_results(results, timestamps, heart_rates, fs):
    """Visualize raw and processed heart amplitude signals (last 60 seconds, English labels)."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互模式
        plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"Font setting error: {e}. English labels may not display correctly.")

    time = np.arange(len(results['raw'])) / fs

    plt.figure(figsize=(15, 14))
    plt.subplot(6, 1, 1)
    plt.plot(time, results['raw'], 'b-', label='Raw Signal')
    plt.title('Raw Heart Amplitude Signal')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.xlim(0, 60)
    plt.xticks(list(plt.gca().get_xticks()) + [60])
    plt.gca().set_xticklabels([str(int(x)) if x == int(x) else f"{x:.1f}" for x in plt.gca().get_xticks()])

    plt.subplot(6, 1, 2)
    plt.plot(time, results['processed']['medfilt'], 'g-', label='Preprocessed')
    plt.title('Detrended and Median Filtered Signal')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.xlim(0, 60)
    plt.xticks(list(plt.gca().get_xticks()) + [60])
    plt.gca().set_xticklabels([str(int(x)) if x == int(x) else f"{x:.1f}" for x in plt.gca().get_xticks()])

    plt.subplot(6, 1, 3)
    plt.plot(time, results['processed']['bandpass'], 'r-', label='Bandpass Filtered')
    plt.title('Bandpass Filtered Signal (0.8-2.0 Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.xlim(0, 60)
    plt.xticks(list(plt.gca().get_xticks()) + [60])
    plt.gca().set_xticklabels([str(int(x)) if x == int(x) else f"{x:.1f}" for x in plt.gca().get_xticks()])

    plt.subplot(6, 1, 4)
    plt.plot(time, results['processed']['wavelet'], 'm-', label='Wavelet Denoised')
    plt.title('Wavelet Denoised Signal')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.xlim(0, 60)
    plt.xticks(list(plt.gca().get_xticks()) + [60])
    plt.gca().set_xticklabels([str(int(x)) if x == int(x) else f"{x:.1f}" for x in plt.gca().get_xticks()])

    plt.subplot(6, 1, 5)
    plt.plot(time, results['processed']['pca'], 'c-', label='PCA Denoised')
    plt.title('PCA Reconstructed Signal')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.xlim(0, 60)
    plt.xticks(list(plt.gca().get_xticks()) + [60])
    plt.gca().set_xticklabels([str(int(x)) if x == int(x) else f"{x:.1f}" for x in plt.gca().get_xticks()])

    plt.xlabel('Time (seconds)')
    plt.tight_layout()
    
    # 保存圖片而不是顯示
    plt.savefig('raw_ecg_vs_bandpass_filtered_ecg.png', dpi=300, bbox_inches='tight')
    print("📊 四種濾波對比圖已儲存為: four_filter_comparison_no_ecg.png")
    plt.close()

# %% [markdown]
# ## 5. Main Function

# %%
def main():
    """Main program entry."""
    file_path = "/path/to/mmwave/data.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        file_path = input("Please enter the full path to the CSV file: ")
    
    waveform_data, timestamps, heart_rates, frame_numbers, fs = load_and_preprocess_data(file_path)
    
    print("\nProcessing heart amplitude data...")
    results = process_radar_heartbeat(waveform_data, fs)
    print("Processing complete!\n")
    
    original_snr = np.mean(results['raw']**2) / np.var(results['raw'])
    processed_snr = np.mean(results['processed']['final']**2) / np.var(results['processed']['final'])
    print(f"SNR improvement: {processed_snr/original_snr:.2f} times")
    
    # 計算並輸出四種濾波與原始信號的比較指標
    comparison_results = calculate_signal_comparisons(results['raw'], results['processed'])
    
    print("\nGenerating visualization results...")
    visualize_results(results, timestamps, heart_rates, fs)
    
    return results, comparison_results

if __name__ == "__main__":
    results, comparison_results = main()
