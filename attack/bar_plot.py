import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from pathlib import Path

plt.rcParams['hatch.color'] = '#555555'
plt.rcParams['hatch.linewidth'] = 1.2

from plot import find_log_files, parse_log_file


# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'STSong']
plt.rcParams['axes.unicode_minus'] = False


def compute_final_accuracy(log_identifiers, log_dir='log_for_plot'):
    """
    计算一组日志的最终准确率。

    log_identifiers: 日志目录名称或路径列表
    返回: 最终准确率的平均值（0-100）
    """
    log_files = find_log_files(log_identifiers, log_dir=log_dir)
    final_accs = []

    for log_file in log_files:
        _, accuracies = parse_log_file(log_file)
        if len(accuracies) == 0:
            continue
        final_acc = accuracies[-1]
        final_accs.append(final_acc)

    if not final_accs:
        return np.nan
    return float(np.mean(final_accs))


def plot_final_accuracy_bar(log_dir='log_for_plot'):
    """
    绘制 34 / 50 / 101 三组 LMm 与 AT 的最终准确率柱状图。
    """
    groups = ['ResNet34', 'ResNet50', 'ResNet101']
    lmm_configs = {
        'ResNet34': ['34LMm1', '34LMm2', '34LMm3'],
        'ResNet50': ['50LMm1', '50LMm2', '50LMm3'],
        'ResNet101': ['101LMm1', '101LMm2', '101LMm3'],
    }
    at_configs = {
        'ResNet34': ['34at1', '34at2', '34at3'],
        'ResNet50': ['50at1', '50at2', '50at3'],
        'ResNet101': ['101at1', '101at2', '101at3'],
    }

    default_clean_accs = {
        'ResNet34': {'lm': 95.6, 'at': 85.0},
        'ResNet50': {'lm': 95.6, 'at': 69.4},
        'ResNet101': {'lm': 95.2, 'at': 66.8},
    }

    lm_clean_accs = []
    lmm_accs = []
    at_clean_accs = []
    at_accs = []
    for group in groups:
        lm_clean_accs.append(default_clean_accs[group]['lm'])
        lmm_accs.append(compute_final_accuracy(lmm_configs[group], log_dir=log_dir))
        at_clean_accs.append(default_clean_accs[group]['at'])
        at_accs.append(compute_final_accuracy(at_configs[group], log_dir=log_dir))

    x = np.arange(len(groups))
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(10, 7))

    base_green = '#7fbf7f'
    base_red = '#ff7f7f'

    lm_clean_color = base_green          # LM 干净（绿色）
    lmm_color = base_red                 # LM+R_sm （红色）
    at_clean_color = base_green
    at_attack_color = base_red

    lm_pastel = '#f3fbf3'
    lmm_pastel = '#fdecec'

    ax.bar(x - 1.5 * bar_width, lm_clean_accs, bar_width, color=lm_pastel, edgecolor=base_green, hatch='///', label='LM' + r'$+R_{\text{sm}}$ (干净)')
    ax.bar(x - 0.5 * bar_width, lmm_accs, bar_width, color=lmm_pastel, edgecolor=base_red, hatch='///', label='LM' + r'$+R_{\text{sm}}$ (攻击)')
    ax.bar(x + 0.5 * bar_width, at_clean_accs, bar_width, color=at_clean_color, edgecolor='none', label='AT (干净)')
    ax.bar(x + 1.5 * bar_width, at_accs, bar_width, color=at_attack_color, edgecolor='none', label='AT (攻击)')

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=28)

    ax.set_xlabel('', fontsize=26)
    ax.set_ylabel('测试准确率 (%)', fontsize=26)
    ax.tick_params(axis='x', which='major', labelsize=24)
    ax.tick_params(axis='x', which='minor', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=22)
    ax.tick_params(axis='y', which='minor', labelsize=18)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=22, loc='upper right', bbox_to_anchor=(1, 0.957))

    # 显示数值标签
    for idx, value in enumerate(lm_clean_accs):
        if np.isnan(value):
            continue
        ax.text(x[idx] - 1.5 * bar_width, value + 0.5, f'{value:.1f}', ha='center', va='bottom', fontsize=20)
    for idx, value in enumerate(lmm_accs):
        if np.isnan(value):
            continue
        ax.text(x[idx] - 0.5 * bar_width, value + 0.5, f'{value:.1f}', ha='center', va='bottom', fontsize=20)
    for idx, value in enumerate(at_clean_accs):
        if np.isnan(value):
            continue
        ax.text(x[idx] + 0.5 * bar_width, value + 0.5, f'{value:.1f}', ha='center', va='bottom', fontsize=20)
    for idx, value in enumerate(at_accs):
        if np.isnan(value):
            continue
        ax.text(x[idx] + 1.5 * bar_width, value + 0.5, f'{value:.1f}', ha='center', va='bottom', fontsize=20)

    # 设置统一的 y 轴范围
    all_values = [v for v in lm_clean_accs + lmm_accs + at_clean_accs + at_accs if not np.isnan(v)]
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min if y_max != y_min else max(1.0, y_max * 0.05)
        y_margin = y_range * 0.1
        lower = max(0, y_min - y_margin)
        upper = min(100, y_max + y_margin)
        ax.set_ylim(lower, upper)

    fig.tight_layout()

    download_dir = Path.home() / 'Downloads'
    download_dir.mkdir(parents=True, exist_ok=True)
    output_path = download_dir / 'resnet_accuracy_bars.pdf'
    fig.savefig(output_path, format='pdf')

    plt.show()


if __name__ == '__main__':
    plot_final_accuracy_bar()


