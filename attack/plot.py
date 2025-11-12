import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 配置中文字体，解决中文渲染问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'STSong']
plt.rcParams['axes.unicode_minus'] = False


def format_legend_label(label: str) -> str:
    """格式化图例标签：去掉数字、移除'w'，并将结尾的'm'替换为 Latex 表示。"""
    formatted = re.sub(r'^[\d-]+', '', label)
    formatted = formatted.replace('w', '')
    if formatted.endswith('m'):
        formatted = formatted[:-1] + r'$+R_{\text{sm}}$'
    return formatted


def parse_log_file(log_path):
    """
    解析日志文件，提取准确率变化趋势
    返回: (iterations, accuracies)
    iterations: 迭代次数列表
    accuracies: 准确率列表（百分比）
    """
    iterations = []
    accuracies = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 跳过第一行（参数信息）
        # 第二行是初始准确率（浮点数）
        if len(lines) >= 2:
            try:
                initial_acc = float(lines[1].strip())
                iterations.append(0)
                accuracies.append(initial_acc * 100)  # 转换为百分比
            except ValueError:
                pass
        
        # 从第三行开始解析迭代记录
        for line in lines[2:]:
            # 匹配格式: "迭代次数: acc=84.00% ..."
            match = re.match(r'(\d+):\s*acc=([\d.]+)%', line)
            if match:
                iteration = int(match.group(1))
                accuracy = float(match.group(2))
                iterations.append(iteration)
                accuracies.append(accuracy)
    
    return np.array(iterations), np.array(accuracies)


def find_log_files(log_identifiers, log_dir='log_for_plot'):
    """
    根据标识符查找日志文件
    log_identifiers: 可以是目录名、完整路径或路径模式
    返回: 日志文件路径列表
    """
    log_files = []
    
    # 如果log_dir不存在，尝试直接使用标识符作为路径
    if not os.path.exists(log_dir):
        log_dir = ''
    
    for identifier in log_identifiers:
        # 如果是完整路径且文件存在
        if os.path.isfile(identifier):
            log_files.append(identifier)
        # 如果是目录路径
        elif os.path.isdir(identifier):
            log_path = os.path.join(identifier, 'log.txt')
            if os.path.exists(log_path):
                log_files.append(log_path)
        # 如果是相对于log_dir的目录名
        elif log_dir and os.path.exists(log_dir):
            # 尝试直接匹配目录名
            log_path = os.path.join(log_dir, identifier, 'log.txt')
            if os.path.exists(log_path):
                log_files.append(log_path)
            else:
                # 尝试部分匹配目录名
                for item in os.listdir(log_dir):
                    if identifier in item:
                        log_path = os.path.join(log_dir, item, 'log.txt')
                        if os.path.exists(log_path):
                            log_files.append(log_path)
                            break
    
    return log_files


def average_logs(log_files):
    """
    对多个日志文件求平均、最大值和最小值
    返回: (iterations, averaged_accuracies, min_accuracies, max_accuracies)
    """
    if not log_files:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # 解析所有日志文件
    all_data = []
    for log_file in log_files:
        if os.path.exists(log_file):
            iterations, accuracies = parse_log_file(log_file)
            if len(iterations) > 0:
                all_data.append((iterations, accuracies))
        else:
            print(f"Warning: File not found: {log_file}")
    
    if not all_data:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # 找到所有出现过的迭代次数
    all_iterations = set()
    for iterations, _ in all_data:
        all_iterations.update(iterations)
    all_iterations = sorted(all_iterations)
    
    # 对每个迭代次数求平均、最小和最大准确率
    averaged_accuracies = []
    min_accuracies = []
    max_accuracies = []
    for it in all_iterations:
        accuracies_at_it = []
        for iterations, accuracies in all_data:
            # 找到该迭代次数对应的准确率
            idx = np.where(iterations == it)[0]
            if len(idx) > 0:
                accuracies_at_it.append(accuracies[idx[0]])
        
        if accuracies_at_it:
            averaged_accuracies.append(np.mean(accuracies_at_it))
            min_accuracies.append(np.min(accuracies_at_it))
            max_accuracies.append(np.max(accuracies_at_it))
        else:
            averaged_accuracies.append(np.nan)
            min_accuracies.append(np.nan)
            max_accuracies.append(np.nan)
    
    return np.array(all_iterations), np.array(averaged_accuracies), np.array(min_accuracies), np.array(max_accuracies)


def plot_curves(curve_configs, log_dir='log_for_plot', labels=None, colors=None, linestyles=None, y_limits=None, save_name=None):
    """
    绘制多条曲线，每条曲线可以是多个日志文件的平均值
    
    参数:
        curve_configs: 列表，每个元素是一个日志标识符列表（用于求平均）
                       例如: [['log1', 'log2'], ['log3', 'log4', 'log5']] 表示两条曲线，
                       第一条是log1和log2的平均，第二条是log3、log4、log5的平均
        log_dir: 日志文件所在目录
        labels: 每条曲线的标签列表（可选），如果为None则自动生成
        colors: 每条曲线的颜色列表（可选），如果为None则使用默认颜色
                可以是颜色名称（如'red', 'blue'）或十六进制颜色码（如'#1f77b4'）
        linestyles: 每条曲线的线型列表（可选），如果为None则使用实线
                   可选值: '-' (实线), '--' (虚线), '-.' (点划线), ':' (点线)
        y_limits: 纵轴范围 (min, max)，若为None则根据数据自动计算
    """
    if not curve_configs:
        print("Error: Curve configuration not specified")
        return
    
    # 为每条曲线查找日志文件并求平均
    data_list = []
    curve_labels = []
    
    for i, config in enumerate(curve_configs):
        if not config:
            print(f"Warning: Curve {i+1} configuration is empty, skipping")
            continue
        
        # 查找日志文件
        log_files = find_log_files(config, log_dir)
        if not log_files:
            print(f"Warning: Curve {i+1} found no log files, skipping")
            continue
        
        # 求平均、最小值和最大值
        iterations, accuracies, min_accuracies, max_accuracies = average_logs(log_files)
        if len(iterations) == 0:
            print(f"Warning: Curve {i+1} data is empty, skipping")
            continue
        
        data_list.append((iterations, accuracies, min_accuracies, max_accuracies))
        
        # 生成标签
        if labels and i < len(labels):
            curve_labels.append(labels[i])
        else:
            # 自动生成标签
            if len(log_files) == 1:
                label = os.path.basename(os.path.dirname(log_files[0]))
            else:
                label = f"Curve {i+1} (avg of {len(log_files)} logs)"
            curve_labels.append(label)
        
        print(f"Curve {i+1} ({curve_labels[-1]}): using {len(log_files)} log files")
    
    if not data_list:
        print("Error: No available data to plot")
        return
    
    # 绘制图形（调整宽高比：横轴压缩，纵轴扩大）
    plt.figure(figsize=(9, 8))
    
    # 默认颜色列表
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 如果没有指定颜色，使用默认颜色
    if colors is None:
        colors = default_colors
    
    # 如果没有指定线型，使用实线
    if linestyles is None:
        linestyles = ['-'] * len(data_list)
    
    for i, data in enumerate(data_list):
        if len(data) == 4:
            iterations, accuracies, min_accuracies, max_accuracies = data
        else:
            # 兼容旧版本（只有iterations和accuracies）
            iterations, accuracies = data
            min_accuracies = None
            max_accuracies = None
        
        # 移除NaN值
        valid_mask = ~np.isnan(accuracies)
        valid_iterations = iterations[valid_mask]
        valid_accuracies = accuracies[valid_mask]
        
        # 将准确率转换为错误率 (100% - accuracy)
        error_rates = 100 - valid_accuracies
        
        # 获取当前曲线的颜色和线型
        curve_color = colors[i] if i < len(colors) else default_colors[i % len(default_colors)]
        curve_linestyle = linestyles[i] if i < len(linestyles) else '-'
        
        # 如果有最小值和最大值，绘制极差范围（半透明区域）
        if min_accuracies is not None and max_accuracies is not None:
            valid_min_mask = ~np.isnan(min_accuracies)
            valid_max_mask = ~np.isnan(max_accuracies)
            valid_range_mask = valid_min_mask & valid_max_mask
            
            if np.any(valid_range_mask):
                valid_range_iterations = iterations[valid_range_mask]
                min_error_rates = 100 - min_accuracies[valid_range_mask]
                max_error_rates = 100 - max_accuracies[valid_range_mask]
                
                # 绘制半透明的极差范围
                plt.fill_between(valid_range_iterations, min_error_rates, max_error_rates,
                               color=curve_color, alpha=0.15, label='_nolegend_')
        
        # 绘制均值曲线
        plt.plot(valid_iterations, error_rates, 
                label=curve_labels[i], 
                color=curve_color,
                linestyle=curve_linestyle,
                linewidth=2.5,
                marker='o' if len(valid_iterations) < 50 else None,
                markersize=4 if len(valid_iterations) < 50 else None,
                alpha=0.8)
    
    plt.xlabel('攻击迭代次数', fontsize=28)
    plt.ylabel('ASR (%)', fontsize=28)
    plt.legend(fontsize=24, loc='best')
    plt.grid(True, alpha=0.3)
    
    # 增大坐标轴刻度标签的字号
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    
    if y_limits is not None:
        plt.ylim(*y_limits)
    else:
        # 自动调整 y 轴范围，让曲线差异更明显
        # 获取所有错误率的最小值和最大值（包括极差范围）
        all_error_rates = []
        for data in data_list:
            if len(data) == 4:
                iterations, accuracies, min_accuracies, max_accuracies = data
                # 包含极差范围
                valid_mask = ~np.isnan(accuracies)
                valid_min_mask = ~np.isnan(min_accuracies) if min_accuracies is not None else np.zeros_like(valid_mask, dtype=bool)
                valid_max_mask = ~np.isnan(max_accuracies) if max_accuracies is not None else np.zeros_like(valid_mask, dtype=bool)
                
                valid_accuracies = accuracies[valid_mask]
                error_rates = 100 - valid_accuracies
                all_error_rates.extend(error_rates)
                
                # 添加最小值和最大值
                if min_accuracies is not None:
                    min_error_rates = 100 - min_accuracies[valid_min_mask]
                    all_error_rates.extend(min_error_rates)
                if max_accuracies is not None:
                    max_error_rates = 100 - max_accuracies[valid_max_mask]
                    all_error_rates.extend(max_error_rates)
            else:
                # 兼容旧版本
                iterations, accuracies = data
                valid_mask = ~np.isnan(accuracies)
                valid_accuracies = accuracies[valid_mask]
                error_rates = 100 - valid_accuracies
                if len(error_rates) > 0:
                    all_error_rates.extend(error_rates)
        
        if all_error_rates:
            y_min = min(all_error_rates)
            y_max = max(all_error_rates)
            y_range = y_max - y_min
            # 在上下各留出10%的边距
            y_margin = y_range * 0.1
            plt.ylim(max(0, y_min - y_margin), min(100, y_max + y_margin))
    
    plt.tight_layout()

    if save_name:
        download_dir = Path.home() / 'Downloads'
        download_dir.mkdir(parents=True, exist_ok=True)
        output_path = download_dir / save_name
        plt.savefig(output_path, format='pdf')

    # 显示图形
    plt.show()


def compute_error_rate_range(group_curve_configs, log_dir='log_for_plot'):
    """
    计算一组曲线配置的统一错误率（ASR）范围
    group_curve_configs: 列表，每个元素是一个curve_configs集合
    返回: (y_min, y_max) 或 None
    """
    all_error_rates = []
    
    for curve_configs in group_curve_configs:
        for config in curve_configs:
            if not config:
                continue
            log_files = find_log_files(config, log_dir)
            if not log_files:
                continue
            iterations, accuracies, min_accuracies, max_accuracies = average_logs(log_files)
            # 收集均值
            if accuracies is not None and len(accuracies) > 0:
                valid = accuracies[~np.isnan(accuracies)]
                if len(valid) > 0:
                    all_error_rates.extend(100 - valid)
            # 收集最小值
            if min_accuracies is not None and len(min_accuracies) > 0:
                valid = min_accuracies[~np.isnan(min_accuracies)]
                if len(valid) > 0:
                    all_error_rates.extend(100 - valid)
            # 收集最大值
            if max_accuracies is not None and len(max_accuracies) > 0:
                valid = max_accuracies[~np.isnan(max_accuracies)]
                if len(valid) > 0:
                    all_error_rates.extend(100 - valid)
    
    if not all_error_rates:
        return None
    
    y_min = min(all_error_rates)
    y_max = max(all_error_rates)
    y_range = y_max - y_min
    if y_range == 0:
        y_margin = max(1.0, y_max * 0.05) if y_max != 0 else 1.0
    else:
        y_margin = y_range * 0.1
    lower = max(0, y_min - y_margin)
    upper = min(100, y_max + y_margin)
    return (lower, upper)

def plot_three_logs(log_dir='log_for_plot'):
    """
    兼容旧版本的函数：从attackLog目录读取三个日志文件并绘制在同一张图上
    """
    # 获取所有日志目录
    log_dirs = []
    if os.path.exists(log_dir):
        for item in os.listdir(log_dir):
            log_path = os.path.join(log_dir, item, 'log.txt')
            if os.path.exists(log_path):
                log_dirs.append(log_path)
    
    if len(log_dirs) < 3:
        print(f"Warning: Only found {len(log_dirs)} log files, need at least 3")
        if len(log_dirs) == 0:
            print("No log files found")
            return
        selected_logs = log_dirs[:len(log_dirs)]
    else:
        selected_logs = sorted(log_dirs, key=lambda x: os.path.getmtime(x), reverse=True)[:3]
    
    # 转换为新的格式：每条曲线对应一个日志文件
    curve_configs = [[os.path.basename(os.path.dirname(log))] for log in selected_logs]
    labels = [os.path.basename(os.path.dirname(log)) for log in selected_logs]
    
    plot_curves(curve_configs, log_dir, labels)


if __name__ == '__main__':
    # 示例1: 绘制三条曲线，每条曲线是单个日志文件
    # plot_three_logs()
    

    # 定义颜色用于方法（CEw, CEm, LMw, LMm, AT）
    method_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 蓝色、橙色、绿色、红色、紫色
    
    # 34组配置
    curve_configs_34 = [
        ['34CEw1', '34CEw2', '34CEw3'],
        ['34CEm1', '34CEm2', '34CEm3'],
        ['34LMw1', '34LMw2', '34LMw3'],
        ['34LMm1', '34LMm2', '34LMm3'],
        ['34at1', '34at2', '34at3']
    ]
    #labels_34 = ['34CEw', '34CEm', '34LMw', '34LMm']
    labels_34 = ['34CEw', '34CEm', '34LMw', '34LMm', '34AT']
    display_labels_34 = [format_legend_label(label) for label in labels_34]
    
    # 50组配置
    curve_configs_50 = [
        ['50CEw1', '50CEw2', '50CEw3'],
        ['50CEm1', '50CEm2', '50CEm3'],
        ['50LMw1', '50LMw2', '50LMw3'],
        ['50LMm1', '50LMm2', '50LMm3'],
        ['50at1', '50at2', '50at3']
    ]
    # labels_50 = ['50CEw', '50CEm', '50LMw', '50LMm']
    labels_50 = ['50CEw', '50CEm', '50LMw', '50LMm', '50AT']
    display_labels_50 = [format_legend_label(label) for label in labels_50]

    # 101组配置
    curve_configs_101 = [
        ['101CEw1', '101CEw2', '101CEw3'],
        ['101CEm1', '101CEm2', '101CEm3'],
        ['101LMw1', '101LMw2', '101LMw3'],
        ['101LMm1', '101LMm2', '101LMm3'],
        ['101at1', '101at2', '101at3']
    ]
    # labels_101 = ['101CEw', '101CEm', '101LMw', '101LMm']
    labels_101 = ['101CEw', '101CEm', '101LMw', '101LMm', '101AT']
    display_labels_101 = [format_legend_label(label) for label in labels_101]

    curve_configs_22_10 = [
        ['22-10CEw1', '22-10CEw2', '22-10CEw3'],
        ['22-10CEm1', '22-10CEm2', '22-10CEm3'],
        ['22-10LMw1', '22-10LMw2', '22-10LMw3'],
        ['22-10LMm1', '22-10LMm2', '22-10LMm3']
    ]
    labels_22_10 = ['22-10CEw', '22-10CEm', '22-10LMw', '22-10LMm']
    display_labels_22_10 = [format_legend_label(label) for label in labels_22_10]

    curve_configs_28_8 = [
        ['28-8CEw1', '28-8CEw2', '28-8CEw3'],
        ['28-8CEm1', '28-8CEm2', '28-8CEm3'],
        ['28-8LMw1', '28-8LMw2', '28-8LMw3'],
        ['28-8LMm1', '28-8LMm2', '28-8LMm3']
    ]
    labels_28_8 = ['28-8CEw', '28-8CEm', '28-8LMw', '28-8LMm']
    display_labels_28_8 = [format_legend_label(label) for label in labels_28_8]


    curve_configs_40_6 = [
        ['40-6CEw1', '40-6CEw2', '40-6CEw3'],
        ['40-6CEm1', '40-6CEm2', '40-6CEm3'],
        ['40-6LMw1', '40-6LMw2', '40-6LMw3'],
        ['40-6LMm1', '40-6LMm2', '40-6LMm3']
    ]
    labels_40_6 = ['40-6CEw', '40-6CEm', '40-6LMw', '40-6LMm']
    display_labels_40_6 = [format_legend_label(label) for label in labels_40_6]

    curve_configs_46_6 = [
        ['46-6CEw1', '46-6CEw2', '46-6CEw3'],
        ['46-6CEm1', '46-6CEm2', '46-6CEm3'],
        ['46-6LMw1', '46-6LMw2', '46-6LMw3'],
        ['46-6LMm1', '46-6LMm2', '46-6LMm3']
    ]
    labels_46_6 = ['46-6CEw', '46-6CEm', '46-6LMw', '46-6LMm']
    display_labels_46_6 = [format_legend_label(label) for label in labels_46_6]

    curve_configs_58_5 = [
        ['58-5CEw1', '58-5CEw2', '58-5CEw3'],
        ['58-5CEm1', '58-5CEm2', '58-5CEm3'],
        ['58-5LMw1', '58-5LMw2', '58-5LMw3'],
        ['58-5LMm1', '58-5LMm2', '58-5LMm3']
    ]
    labels_58_5 = ['58-5CEw', '58-5CEm', '58-5LMw', '58-5LMm']
    display_labels_58_5 = [format_legend_label(label) for label in labels_58_5]

    curve_configs_82_4 = [
        ['82-4CEw1', '82-4CEw2', '82-4CEw3'],
        ['82-4CEm1', '82-4CEm2', '82-4CEm3'],
        ['82-4LMw1', '82-4LMw2', '82-4LMw3'],
        ['82-4LMm1', '82-4LMm2', '82-4LMm3']
    ]
    labels_82_4 = ['82-4CEw', '82-4CEm', '82-4LMw', '82-4LMm']
    display_labels_82_4 = [format_legend_label(label) for label in labels_82_4]



    # # 计算统一的纵轴范围
    # global_y_limits = compute_error_rate_range(
    #     [curve_configs_34, curve_configs_50, curve_configs_101],
    #     log_dir='log_for_plot'
    # )

    # 计算统一的纵轴范围
    global_y_limits = compute_error_rate_range(
        [curve_configs_22_10, curve_configs_28_8, curve_configs_40_6,
         curve_configs_46_6, curve_configs_58_5, curve_configs_82_4],
        log_dir='log_for_plot'
    )

    # 依次绘制三组图像
    # plot_curves(curve_configs_34, log_dir='log_for_plot', labels=display_labels_34, colors=method_colors, y_limits=global_y_limits, save_name='resnet34_curves.pdf')
    # plot_curves(curve_configs_50, log_dir='log_for_plot', labels=display_labels_50, colors=method_colors, y_limits=global_y_limits, save_name='resnet50_curves.pdf')
    # plot_curves(curve_configs_101, log_dir='log_for_plot', labels=display_labels_101, colors=method_colors, y_limits=global_y_limits, save_name='resnet101_curves.pdf')

    plot_curves(curve_configs_22_10, log_dir='log_for_plot', labels=display_labels_22_10, colors=method_colors,
                y_limits=global_y_limits, save_name='WRN22-10_curves.pdf')
    plot_curves(curve_configs_28_8, log_dir='log_for_plot', labels=display_labels_28_8, colors=method_colors,
                y_limits=global_y_limits, save_name='WRN28-8_curves.pdf')
    plot_curves(curve_configs_40_6, log_dir='log_for_plot', labels=display_labels_40_6, colors=method_colors,
                y_limits=global_y_limits, save_name='WRN40-6_curves.pdf')
    plot_curves(curve_configs_46_6, log_dir='log_for_plot', labels=display_labels_46_6, colors=method_colors,
                y_limits=global_y_limits, save_name='WRN46-6_curves.pdf')
    plot_curves(curve_configs_58_5, log_dir='log_for_plot', labels=display_labels_58_5, colors=method_colors,
                y_limits=global_y_limits, save_name='WRN58-5_curves.pdf')
    plot_curves(curve_configs_82_4, log_dir='log_for_plot', labels=display_labels_82_4, colors=method_colors,
                y_limits=global_y_limits, save_name='WRN82-4_curves.pdf')






















