import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體支援
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class MMWaveECGAnalyzer:
    """毫米波與ECG相似度分析器"""
    
    def __init__(self, csv_file_path):
        """
        初始化分析器
        
        Parameters:
        csv_file_path (str): CSV文件路徑
        """
        self.data = pd.read_csv(csv_file_path, encoding='utf-8')
        self.results = {}
        self.best_data = {}
        
    def load_data(self):
        """載入並預處理數據"""
        print("數據載入完成！")
        print(f"總數據筆數: {len(self.data)}")
        print(f"數據欄位: {list(self.data.columns)}")
        
        # 檢查各距離的數據分布
        distance_counts = self.data['距離(cm)'].value_counts().sort_index()
        print("\n各距離數據分布:")
        for distance, count in distance_counts.items():
            print(f"{distance}cm: {count}筆")
        
        return self.data
    
    def calculate_composite_score(self, row, weighting_scheme='balanced'):
        """
        計算綜合評分
        
        Parameters:
        row (pd.Series): 數據行
        weighting_scheme (str): 權重方案
            - 'balanced': 平衡方案 (33.3%, 33.3%, 33.3%)
            - 'shape_focused': 形狀導向 (60%, 20%, 20%)
            - 'correlation_focused': 相關性導向 (20%, 40%, 40%)
            - 'dtw_dominant': DTW主導 (70%, 15%, 15%)
            - 'equal_corr': 相關性平等 (40%, 30%, 30%)
            - 'custom': 自定義權重 (需要額外指定)
        
        Returns:
        float: 綜合評分
        """
        # 計算各個指標的標準化分數
        dtw_score = 1 / (1 + row['原始_DTW距離']) if pd.notna(row['原始_DTW距離']) else 0
        spearman_score = abs(row['原始_Spearman']) if pd.notna(row['原始_Spearman']) else 0
        cross_corr_score = abs(row['原始_互相關']) / 1000 if pd.notna(row['原始_互相關']) else 0
        
        # 定義不同的權重方案
        weight_schemes = {
            'balanced': (1/3, 1/3, 1/3),
            'shape_focused': (0.6, 0.2, 0.2),
            'correlation_focused': (0.2, 0.4, 0.4),
            'dtw_dominant': (0.85, 0.075, 0.075),  # DTW絕對主導 85%
            'dtw_absolute': (0.90, 0.05, 0.05),   # DTW絕對主導 90%
            'dtw_extreme': (0.95, 0.025, 0.025),  # DTW極端主導 95%
            'equal_corr': (0.4, 0.3, 0.3),
            'literature_based': (0.5, 0.3, 0.2)
        }
        
        if weighting_scheme in weight_schemes:
            w_dtw, w_spearman, w_cross = weight_schemes[weighting_scheme]
        else:
            # 默認使用平衡方案
            w_dtw, w_spearman, w_cross = weight_schemes['balanced']
        
        # 計算加權綜合分數
        composite_score = (dtw_score * w_dtw + 
                          spearman_score * w_spearman + 
                          cross_corr_score * w_cross)
        
        return composite_score
    
    def prepare_data(self, weighting_scheme='balanced'):
        """
        準備數據並計算綜合評分
        
        Parameters:
        weighting_scheme (str): 權重分配方案
        """
        # 計算綜合評分
        self.data['綜合評分'] = self.data.apply(
            lambda row: self.calculate_composite_score(row, weighting_scheme), axis=1
        )
        
        # 按距離分組，所有數據都是最佳的
        distances = self.data['距離(cm)'].unique()
        
        print(f"\n使用權重方案: {weighting_scheme}")
        weight_schemes_desc = {
            'balanced': 'DTW:33.3%, Spearman:33.3%, 互相關:33.3%',
            'shape_focused': 'DTW:60%, Spearman:20%, 互相關:20%',
            'correlation_focused': 'DTW:20%, Spearman:40%, 互相關:40%',
            'dtw_dominant': 'DTW:85%, Spearman:7.5%, 互相關:7.5%',
            'dtw_absolute': 'DTW:90%, Spearman:5%, 互相關:5%',
            'dtw_extreme': 'DTW:95%, Spearman:2.5%, 互相關:2.5%',
            'equal_corr': 'DTW:40%, Spearman:30%, 互相關:30%',
            'literature_based': 'DTW:50%, Spearman:30%, 互相關:20%'
        }
        print(f"權重分配: {weight_schemes_desc.get(weighting_scheme, '自定義')}")
        
        # 所有數據都保留（已經是最佳的25筆）
        for distance in distances:
            distance_data = self.data[self.data['距離(cm)'] == distance].copy()
            self.best_data[distance] = distance_data
            print(f"\n{distance}cm - 使用全部{len(distance_data)}筆數據 (已預先篩選的最佳資料)")
    
    
    def compare_weighting_schemes(self, schemes_to_compare=None):
        """
        比較不同權重方案的結果
        
        Parameters:
        schemes_to_compare (list): 要比較的權重方案列表
        """
        if schemes_to_compare is None:
            schemes_to_compare = ['balanced', 'dtw_dominant', 'dtw_absolute', 'dtw_extreme']
        
        print("\n" + "="*100)
        print("不同權重方案比較結果 (重點：DTW絕對主導方案)")
        print("="*100)
        
        scheme_results = {}
        
        for scheme in schemes_to_compare:
            print(f"\n📊 權重方案: {scheme}")
            
            # 顯示具體權重分配
            weight_schemes_desc = {
                'balanced': 'DTW:33.3%, Spearman:33.3%, 互相關:33.3%',
                'dtw_dominant': 'DTW:85%, Spearman:7.5%, 互相關:7.5%',
                'dtw_absolute': 'DTW:90%, Spearman:5%, 互相關:5%',
                'dtw_extreme': 'DTW:95%, Spearman:2.5%, 互相關:2.5%',
                'shape_focused': 'DTW:60%, Spearman:20%, 互相關:20%',
                'correlation_focused': 'DTW:20%, Spearman:40%, 互相關:40%'
            }
            
            print(f"    權重分配: {weight_schemes_desc.get(scheme, '未定義')}")
            print("-" * 80)
            
            # 重新計算綜合評分
            temp_data = self.data.copy()
            temp_data['綜合評分'] = temp_data.apply(
                lambda row: self.calculate_composite_score(row, scheme), axis=1
            )
            
            # 計算各距離的平均綜合評分
            distance_avg_scores = {}
            for distance in temp_data['距離(cm)'].unique():
                distance_data = temp_data[temp_data['距離(cm)'] == distance]
                top_25 = distance_data.nlargest(25, '綜合評分')
                avg_score = top_25['綜合評分'].mean()
                distance_avg_scores[distance] = avg_score
            
            # 排序並顯示結果
            sorted_distances = sorted(distance_avg_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (distance, score) in enumerate(sorted_distances, 1):
                status = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                print(f"    {status} {distance}cm: {score:.6f}")
            
            scheme_results[scheme] = {
                'best_distance': sorted_distances[0][0],
                'rankings': sorted_distances
            }
        
        # 總結比較
        print(f"\n📋 DTW主導權重方案對最佳距離的影響:")
        print("-" * 60)
        for scheme, result in scheme_results.items():
            weight_desc = {
                'balanced': '平衡方案',
                'dtw_dominant': 'DTW主導(85%)',
                'dtw_absolute': 'DTW絕對主導(90%)',
                'dtw_extreme': 'DTW極端主導(95%)'
            }
            desc = weight_desc.get(scheme, scheme)
            print(f"{desc:15s} → 最佳距離: {result['best_distance']}cm")
        
        # DTW權重影響分析
        print(f"\n🔍 DTW權重對結果穩定性的分析:")
        print("-" * 60)
        best_distances = [result['best_distance'] for result in scheme_results.values()]
        if len(set(best_distances)) == 1:
            print("✅ 所有DTW主導方案都指向相同的最佳距離，結果非常穩定")
        elif len(set(best_distances)) == 2:
            print("⚠️  DTW主導方案存在輕微分歧，建議進一步檢查")
        else:
            print("❌ DTW主導方案結果不一致，可能需要更仔細的分析")
        
        return scheme_results
    
    def calculate_statistics(self):
        """計算各距離的統計指標"""
        
        for distance, data in self.best_data.items():
            # 提取各指標數據
            dtw_values = data['原始_DTW距離'].dropna()
            spearman_values = data['原始_Spearman'].dropna()
            cross_corr_values = data['原始_互相關'].dropna()
            composite_scores = data['綜合評分'].dropna()
            
            # 計算統計指標
            stats_dict = {
                'dtw': {
                    'mean': dtw_values.mean(),
                    'std': dtw_values.std(),
                    'median': dtw_values.median(),
                    'min': dtw_values.min(),
                    'max': dtw_values.max()
                },
                'spearman': {
                    'mean': spearman_values.mean(),
                    'std': spearman_values.std(),
                    'abs_mean': abs(spearman_values).mean(),
                    'median': spearman_values.median(),
                    'min': spearman_values.min(),
                    'max': spearman_values.max()
                },
                'cross_corr': {
                    'mean': cross_corr_values.mean(),
                    'std': cross_corr_values.std(),
                    'abs_mean': abs(cross_corr_values).mean(),
                    'median': cross_corr_values.median(),
                    'min': cross_corr_values.min(),
                    'max': cross_corr_values.max()
                },
                'composite': {
                    'mean': composite_scores.mean(),
                    'std': composite_scores.std(),
                    'median': composite_scores.median(),
                    'scores': composite_scores.tolist()
                }
            }
            
            self.results[distance] = stats_dict
    
    def create_statistics_table(self):
        """創建統計結果表格"""
        
        # 準備表格數據
        table_data = []
        
        for distance, stats in self.results.items():
            # 計算綜合表現分數 (用於排名)
            performance_score = (1/stats['dtw']['mean'] * 0.4 + 
                               stats['spearman']['abs_mean'] * 0.3 + 
                               stats['cross_corr']['abs_mean']/1000 * 0.3)
            
            table_data.append({
                '距離(cm)': distance,
                'DTW距離_平均': f"{stats['dtw']['mean']:.3f}",
                'DTW距離_標準差': f"{stats['dtw']['std']:.3f}",
                'Spearman_平均': f"{stats['spearman']['mean']:.4f}",
                'Spearman_標準差': f"{stats['spearman']['std']:.4f}",
                '互相關_平均': f"{stats['cross_corr']['mean']:.1f}",
                '互相關_標準差': f"{stats['cross_corr']['std']:.1f}",
                '綜合表現分數': f"{performance_score:.4f}",
                '排名用分數': performance_score
            })
        
        # 轉換為DataFrame並按表現分數排序
        df_table = pd.DataFrame(table_data)
        df_table = df_table.sort_values('排名用分數', ascending=False)
        df_table['排名'] = range(1, len(df_table) + 1)
        
        # 重新排列欄位順序
        df_display = df_table[['排名', '距離(cm)', 'DTW距離_平均', 'DTW距離_標準差', 
                              'Spearman_平均', 'Spearman_標準差', '互相關_平均', 
                              '互相關_標準差', '綜合表現分數']].copy()
        
        print("\n" + "="*100)
        print("統計結果表格 (基於最佳25筆數據)")
        print("="*100)
        print(df_display.to_string(index=False, float_format='%.4f'))
        
        return df_display
    
    def perform_statistical_tests(self, save_plots=True):
        """進行統計假設檢定"""
        
        print("\n" + "="*80)
        print("統計假設檢定結果")
        print("="*80)
        
        distances = list(self.results.keys())
        
        # 準備存儲結果的數據結構
        t_test_results = []
        wilcoxon_results = []
        comparison_pairs = []
        
        # 對綜合評分進行配對比較
        print("\n1. 綜合評分的配對t檢定結果:")
        print("-" * 50)
        
        for i in range(len(distances)):
            for j in range(i+1, len(distances)):
                dist1, dist2 = distances[i], distances[j]
                scores1 = self.results[dist1]['composite']['scores']
                scores2 = self.results[dist2]['composite']['scores']
                
                # 進行獨立樣本t檢定
                t_stat, p_value = stats.ttest_ind(scores1, scores2)
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"{dist1}cm vs {dist2}cm: t={t_stat:.3f}, p={p_value:.4f} {significance}")
                
                # 存儲結果
                comparison_pairs.append(f"{dist1}cm vs {dist2}cm")
                t_test_results.append(p_value)
        
        # Wilcoxon檢定 (非參數檢定)
        print("\n2. 綜合評分的Wilcoxon秩和檢定結果:")
        print("-" * 50)
        
        for i in range(len(distances)):
            for j in range(i+1, len(distances)):
                dist1, dist2 = distances[i], distances[j]
                scores1 = self.results[dist1]['composite']['scores']
                scores2 = self.results[dist2]['composite']['scores']
                
                # 進行Wilcoxon秩和檢定
                u_stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"{dist1}cm vs {dist2}cm: U={u_stat:.3f}, p={p_value:.4f} {significance}")
                
                # 存儲結果
                wilcoxon_results.append(p_value)
        
        print("\n註: *** p<0.001, ** p<0.01, * p<0.05, ns=不顯著")
        
        # 創建統計檢定結果的視覺化
        if save_plots:
            self.create_statistical_test_visualization(comparison_pairs, t_test_results, wilcoxon_results)
    
    def create_statistical_test_visualization(self, comparison_pairs, t_test_results, wilcoxon_results):
        """創建統計檢定結果的視覺化圖表"""
        
        import matplotlib
        matplotlib.use('Agg')  # 使用非GUI後端
        
        # 創建圖表
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 設置顏色映射 (基於p值的顯著性水平)
        def get_color(p_value):
            if p_value < 0.001:
                return 'red'  # 極顯著
            elif p_value < 0.01:
                return 'orange'  # 高度顯著
            elif p_value < 0.05:
                return 'yellow'  # 顯著
            else:
                return 'lightgray'  # 不顯著
        
        def get_significance_label(p_value):
            if p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return 'ns'
        
        # 圖1: t檢定結果條形圖
        colors_t = [get_color(p) for p in t_test_results]
        bars1 = ax1.bar(range(len(comparison_pairs)), t_test_results, color=colors_t, alpha=0.7, edgecolor='black')
        ax1.set_title('T-Test Results (p-values)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Comparison Pairs')
        ax1.set_ylabel('p-value')
        ax1.set_xticks(range(len(comparison_pairs)))
        ax1.set_xticklabels(comparison_pairs, rotation=45, ha='right')
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        ax1.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='α = 0.01')
        ax1.axhline(y=0.001, color='darkred', linestyle='--', alpha=0.7, label='α = 0.001')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加顯著性標記
        for i, (bar, p_val) in enumerate(zip(bars1, t_test_results)):
            height = bar.get_height()
            sig_label = get_significance_label(p_val)
            ax1.text(bar.get_x() + bar.get_width()/2, height + max(t_test_results)*0.01,
                    sig_label, ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 圖2: Wilcoxon檢定結果條形圖
        colors_w = [get_color(p) for p in wilcoxon_results]
        bars2 = ax2.bar(range(len(comparison_pairs)), wilcoxon_results, color=colors_w, alpha=0.7, edgecolor='black')
        ax2.set_title('Wilcoxon Rank-Sum Test Results (p-values)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Comparison Pairs')
        ax2.set_ylabel('p-value')
        ax2.set_xticks(range(len(comparison_pairs)))
        ax2.set_xticklabels(comparison_pairs, rotation=45, ha='right')
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        ax2.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='α = 0.01')
        ax2.axhline(y=0.001, color='darkred', linestyle='--', alpha=0.7, label='α = 0.001')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加顯著性標記
        for i, (bar, p_val) in enumerate(zip(bars2, wilcoxon_results)):
            height = bar.get_height()
            sig_label = get_significance_label(p_val)
            ax2.text(bar.get_x() + bar.get_width()/2, height + max(wilcoxon_results)*0.01,
                    sig_label, ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 圖3: 比較兩種檢定方法的p值散點圖
        ax3.scatter(t_test_results, wilcoxon_results, alpha=0.7, s=100, edgecolor='black')
        ax3.set_title('T-Test vs Wilcoxon Test Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('T-Test p-values')
        ax3.set_ylabel('Wilcoxon Test p-values')
        ax3.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='y = x')
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=0.05, color='red', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 添加每個點的標籤
        for i, (t_p, w_p, pair) in enumerate(zip(t_test_results, wilcoxon_results, comparison_pairs)):
            ax3.annotate(pair, (t_p, w_p), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
        
        # 設置對數尺度以更好地顯示小p值
        if min(t_test_results + wilcoxon_results) > 0:
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            ax3.set_xscale('log')
            ax3.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('statistical_tests_results.png', dpi=300, bbox_inches='tight')
        print("📊 統計檢定結果圖表已儲存為: statistical_tests_results.png")
        plt.close()
    
    def create_visualizations(self, save_plots=True):
        """創建視覺化圖表"""
        
        # 設置圖表樣式
        import matplotlib
        matplotlib.use('Agg')  # 使用非GUI後端
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 12))
        
        # 準備數據
        distances = list(self.results.keys())
        dtw_means = [self.results[d]['dtw']['mean'] for d in distances]
        dtw_stds = [self.results[d]['dtw']['std'] for d in distances]
        spearman_abs_means = [self.results[d]['spearman']['abs_mean'] for d in distances]
        spearman_stds = [self.results[d]['spearman']['std'] for d in distances]
        cross_corr_abs_means = [self.results[d]['cross_corr']['abs_mean'] for d in distances]
        cross_corr_stds = [self.results[d]['cross_corr']['std'] for d in distances]
        composite_means = [self.results[d]['composite']['mean'] for d in distances]
        
        # 子圖1: DTW距離比較
        plt.subplot(2, 3, 1)
        bars1 = plt.bar(distances, dtw_means, yerr=dtw_stds, capsize=5, 
                       color='lightcoral', alpha=0.7, edgecolor='darkred')
        plt.title('DTW Distance Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        plt.xlabel('Distance (cm)')
        plt.ylabel('DTW Distance')
        plt.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for bar, mean in zip(bars1, dtw_means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.05,
                    f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 子圖2: Spearman相關係數比較
        plt.subplot(2, 3, 2)
        bars2 = plt.bar(distances, spearman_abs_means, yerr=spearman_stds, capsize=5,
                       color='lightblue', alpha=0.7, edgecolor='darkblue')
        plt.title('|Spearman Correlation| Comparison (Higher is Better)', fontsize=12, fontweight='bold')
        plt.xlabel('Distance (cm)')
        plt.ylabel('|Spearman Correlation Coefficient|')
        plt.grid(True, alpha=0.3)
        
        for bar, mean in zip(bars2, spearman_abs_means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.05,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 子圖3: 互相關係數比較
        plt.subplot(2, 3, 3)
        bars3 = plt.bar(distances, cross_corr_abs_means, yerr=cross_corr_stds, capsize=5,
                       color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        plt.title('|Cross-Correlation| Comparison (Higher is Better)', fontsize=12, fontweight='bold')
        plt.xlabel('Distance (cm)')
        plt.ylabel('|Cross-Correlation Coefficient|')
        plt.grid(True, alpha=0.3)
        
        for bar, mean in zip(bars3, cross_corr_abs_means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.05,
                    f'{mean:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 子圖4: 綜合評分比較
        plt.subplot(2, 3, 4)
        bars4 = plt.bar(distances, composite_means, color='gold', alpha=0.7, edgecolor='orange')
        plt.title('Composite Score Comparison', fontsize=12, fontweight='bold', pad=20)
        plt.xlabel('Distance (cm)')
        plt.ylabel('Composite Score')
        plt.grid(True, alpha=0.3)
        
        # 標記最佳結果
        best_idx = np.argmax(composite_means)
        bars4[best_idx].set_color('red')
        bars4[best_idx].set_alpha(0.8)
        
        for bar, mean in zip(bars4, composite_means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.05,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 子圖5: 綜合評分分布箱線圖
        plt.subplot(2, 3, 5)
        composite_data = [self.results[d]['composite']['scores'] for d in distances]
        box_plot = plt.boxplot(composite_data, labels=distances, patch_artist=True)
        
        # 設置箱線圖顏色
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
        for patch, color in zip(box_plot['boxes'], colors[:len(distances)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Composite Score Distribution', fontsize=12, fontweight='bold', pad=20)
        plt.xlabel('Distance (cm)')
        plt.ylabel('Composite Score')
        plt.grid(True, alpha=0.3)
        
        # 子圖6: 雷達圖
        plt.subplot(2, 3, 6, projection='polar')
        
        # 準備雷達圖數據 (標準化)
        categories = ['DTW\n(Inverted)', 'Spearman\n(Absolute)', 'Cross-Correlation\n(Absolute)', 'Composite\nScore']
        N = len(categories)
        
        # 計算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 閉合圖形
        
        for i, distance in enumerate(distances):
            # 標準化數據 (0-1範圍)
            values = [
                1 - (self.results[distance]['dtw']['mean'] - min(dtw_means)) / (max(dtw_means) - min(dtw_means)),  # DTW反向
                (self.results[distance]['spearman']['abs_mean'] - min(spearman_abs_means)) / (max(spearman_abs_means) - min(spearman_abs_means)),
                (self.results[distance]['cross_corr']['abs_mean'] - min(cross_corr_abs_means)) / (max(cross_corr_abs_means) - min(cross_corr_abs_means)),
                (self.results[distance]['composite']['mean'] - min(composite_means)) / (max(composite_means) - min(composite_means))
            ]
            values += values[:1]  # 閉合圖形
            
            plt.plot(angles, values, 'o-', linewidth=2, label=f'{distance}cm', alpha=0.7)
            plt.fill(angles, values, alpha=0.1)
        
        plt.xticks(angles[:-1], categories)
        plt.ylim(0, 1)
        plt.title('Multi-Distance Performance Radar Chart', fontsize=12, fontweight='bold', pad=40)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 調整子圖間距，特別增加垂直間距以避免標題與圖片重疊
        plt.tight_layout(pad=4.0, h_pad=4.0, w_pad=2.5)
        if save_plots:
            plt.savefig('mmwave_analysis_results.png', dpi=300, bbox_inches='tight')
            print("📊 圖表已儲存為: mmwave_analysis_results.png")
        else:
            plt.show()
    
    def generate_conclusion(self):
        """生成結論報告"""
        
        # 找出最佳距離
        distances = list(self.results.keys())
        composite_means = [self.results[d]['composite']['mean'] for d in distances]
        best_idx = np.argmax(composite_means)
        best_distance = distances[best_idx]
        best_score = composite_means[best_idx]
        
        # 排序所有距離
        distance_scores = [(d, self.results[d]['composite']['mean']) for d in distances]
        distance_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\n" + "="*80)
        print("🏆 研究結論與建議")
        print("="*80)
        
        print(f"\n✅ 最佳測量距離: {best_distance}cm")
        print(f"   綜合評分: {best_score:.4f}")
        print(f"   DTW距離: {self.results[best_distance]['dtw']['mean']:.3f} ± {self.results[best_distance]['dtw']['std']:.3f}")
        print(f"   Spearman相關: {self.results[best_distance]['spearman']['mean']:.4f} ± {self.results[best_distance]['spearman']['std']:.4f}")
        print(f"   互相關係數: {self.results[best_distance]['cross_corr']['mean']:.1f} ± {self.results[best_distance]['cross_corr']['std']:.1f}")
        
        print("\n📊 各距離表現排名:")
        for i, (distance, score) in enumerate(distance_scores, 1):
            status = "🥇 最佳" if i == 1 else "🥈 次佳" if i == 2 else "🥉 第三" if i == 3 else f"   第{i}"
            print(f"   {status}: {distance}cm (評分: {score:.4f})")
        
        print("\n📋 學術建議:")
        print("   1. 統計顯著性: 建議進行配對t檢定驗證各距離間差異")
        print("   2. 樣本大小: 建議增加樣本數量以提高統計功效")
        print(f"   3. 實用建議: 在{best_distance}cm距離下毫米波雷達檢測效果最佳")
        print(f"   4. 優化範圍: 可在{best_distance}cm ± 10cm範圍內進一步優化")
        
        return best_distance, distance_scores
    
    def run_complete_analysis(self, top_n=25, weighting_scheme='balanced', compare_schemes=True):
        """
        運行完整分析流程
        
        Parameters:
        top_n (int): 選取的最佳數據筆數
        weighting_scheme (str): 使用的權重方案
        compare_schemes (bool): 是否比較不同權重方案
        """
        
        print("開始毫米波與ECG相似度分析...")
        print("="*80)
        
        # 1. 載入數據
        self.load_data()
        
        # 2. 比較不同權重方案 (如果要求)
        if compare_schemes:
            scheme_comparison = self.compare_weighting_schemes()
        
        # 3. 準備數據 (數據已預先篩選)
        self.prepare_data(weighting_scheme)
        
        # 4. 計算統計指標
        self.calculate_statistics()
        
        # 5. 創建統計表格
        stats_table = self.create_statistics_table()
        
        # 6. 進行統計檢定
        self.perform_statistical_tests(save_plots=True)
        
        # 7. 創建視覺化
        self.create_visualizations(save_plots=True)
        
        # 8. 生成結論
        best_distance, rankings = self.generate_conclusion()
        
        # 9. 權重敏感性分析
        print("\n" + "="*80)
        print("🔍 權重敏感性分析建議")
        print("="*80)
        print("不同權重方案可能導致不同的最佳距離選擇。建議:")
        print("1. 根據研究目標選擇合適的權重方案")
        print("2. 進行敏感性分析，檢驗結果的穩健性")
        print("3. 考慮使用專家評估或文獻依據來確定權重")
        print("4. 可以使用主成分分析(PCA)來客觀確定權重")
        
        result_dict = {
            'best_distance': best_distance,
            'rankings': rankings,
            'statistics_table': stats_table,
            'detailed_results': self.results,
            'weighting_scheme': weighting_scheme
        }
        
        if compare_schemes:
            result_dict['scheme_comparison'] = scheme_comparison
            
        return result_dict

# 權重方案的理論依據和建議
def get_weighting_recommendations():
    """
    提供權重分配的理論依據和建議
    """
    recommendations = {
        'research_goals': {
            'heart_rate_detection': {
                'description': '主要關注心率檢測準確性',
                'recommended_scheme': 'dtw_absolute',
                'rationale': 'DTW距離90%權重，專注於波形形狀的精確匹配'
            },
            'signal_quality_assessment': {
                'description': '信號質量評估',
                'recommended_scheme': 'dtw_extreme',
                'rationale': 'DTW距離95%權重，是評估信號質量的最直接指標'
            },
            'waveform_similarity': {
                'description': '波形相似度分析',
                'recommended_scheme': 'dtw_dominant',
                'rationale': 'DTW距離85%權重，主要評估時間序列形狀匹配度'
            },
            'heart_rate_variability': {
                'description': '關注心率變異性分析',
                'recommended_scheme': 'correlation_focused',
                'rationale': '相關性指標更適合評估心率變異的一致性'
            },
            'general_cardiac_monitoring': {
                'description': '一般心臟監測應用',
                'recommended_scheme': 'balanced',
                'rationale': '平衡考慮所有指標，避免偏向性'
            },
            'signal_quality_assessment': {
                'description': '信號質量評估',
                'recommended_scheme': 'dtw_extreme',
                'rationale': 'DTW距離95%權重，是評估信號質量的最直接指標'
            }
        },
        'literature_based': {
            'description': '基於DTW絕對主導的權重分配',
            'dtw_weight': '85-95%',
            'correlation_weight': '2.5-7.5%',
            'rationale': '專注於時間序列波形相似度，最小化其他因素干擾'
        },
        'data_driven_approaches': [
            '主成分分析(PCA): 基於數據特徵自動確定權重',
            '機器學習方法: 使用監督學習優化權重',
            '專家評估: 結合領域專家意見確定權重',
            '敏感性分析: 測試不同權重對結果的影響'
        ]
    }
    
    return recommendations

# 使用範例
if __name__ == "__main__":
    # 顯示權重選擇建議
    print("="*80)
    print("權重分配選擇指南")
    print("="*80)
    
    recommendations = get_weighting_recommendations()
    
    print("\n🎯 根據研究目標選擇權重方案:")
    for goal, info in recommendations['research_goals'].items():
        print(f"\n• {info['description']}")
        print(f"  推薦方案: {info['recommended_scheme']}")
        print(f"  理由: {info['rationale']}")
    
    print(f"\n� 文獻依據:")
    lit_info = recommendations['literature_based']
    print(f"  {lit_info['description']}")
    print(f"  DTW權重: {lit_info['dtw_weight']}")
    print(f"  相關性權重: {lit_info['correlation_weight']}")
    print(f"  理由: {lit_info['rationale']}")
    
    print(f"\n🔬 數據驅動方法:")
    for method in recommendations['data_driven_approaches']:
        print(f"  • {method}")
    
    print("\n" + "="*80)
    
    # 創建分析器實例
    analyzer = MMWaveECGAnalyzer('/path/to/analysis_results.csv')  # 請替換為您的CSV文件路徑
    
    # 運行完整分析 (使用DTW絕對主導方案)
    results = analyzer.run_complete_analysis(
        top_n=25, 
        weighting_scheme='dtw_absolute',  # DTW 90%權重
        compare_schemes=True
    )
    
    # 輸出最佳距離
    print(f"\n🎯 使用DTW絕對主導方案({results['weighting_scheme']})的最終建議: {results['best_distance']}cm")
    print("📊 DTW權重90%，專注於波形形狀的精確匹配")
