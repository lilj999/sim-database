import matplotlib.pyplot as plt
import pandas as pd

def plot_hamming_distances_histogram(data_ids, distances, threat_levels, sample_weights, retrieved_weights):
    """
    绘制汉明距离的直方图。
    颜色表示 Threat_Level，细微颜色变化表示权重映射的区间，并在图例中展示所有颜色对应关系。

    :param data_ids: 文档 ID 列表。
    :param distances: 对应的汉明距离列表。
    :param threat_levels: Threat_Level 对应类别列表。
    :param sample_weights: 样本数据的权重映射（字典）。
    :param retrieved_weights: 检索到的数据的权重映射（列表，每个元素为一个字典）。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches

    # 基础颜色映射（Threat_Level）
    base_colors = {
        "Low": "#88CC88",     # 浅绿色
        "Medium": "#FFCC88",  # 浅橙色
        "High": "#FF8888",    # 浅红色
    }

    # 根据权重重合度对每种 Threat_Level 的颜色进行微调
    color_variations = {
        "Low": ["#88CC88", "#66AA66", "#448844"],      # 低威胁：绿色深浅变化
        "Medium": ["#FFCC88", "#EEAA66", "#DD8844"],   # 中威胁：橙色深浅变化
        "High": ["#FF8888", "#DD6666", "#BB4444"],     # 高威胁：红色深浅变化
    }

    # Threat_Level 未知时的处理
    threat_levels = [t if t in base_colors else "Unknown" for t in threat_levels]

    # 确保汉明距离有最小高度
    min_distance = 0.5
    distances = [max(d, min_distance) for d in distances]

    # 按汉明距离排序
    sorted_data = sorted(zip(data_ids, distances, threat_levels), key=lambda x: x[1])
    sorted_ids, sorted_distances, sorted_threat_levels = zip(*sorted_data)

    # 计算权重映射重合度
    overlap_percentages = []
    for r_weight in retrieved_weights:
        overlap_count = len(set(r_weight.keys()) & set(sample_weights.keys()))
        total_count = len(set(r_weight.keys()) | set(sample_weights.keys()))
        overlap_percentages.append((overlap_count / total_count) * 100 if total_count > 0 else 0)

    # 根据权重映射重合度划分区间
    overlap_bins = [0, 40, 60, 100]
    overlap_categories = np.digitize(overlap_percentages, bins=overlap_bins, right=True)

    # 绘制汉明距离直方图
    plt.figure(figsize=(14, 8))
    for idx, (doc_id, distance, threat_level) in enumerate(zip(sorted_ids, sorted_distances, sorted_threat_levels)):
        # 获取对应的颜色
        color_idx = overlap_categories[idx] - 1  # 区间对应索引
        color = color_variations[threat_level][color_idx]

        plt.bar(
            idx,
            distance,
            color=color,
            edgecolor="black",
        )

    # 设置 X 轴和标题
    plt.xticks(range(len(sorted_ids)), sorted_ids, rotation=90, fontsize=8)
    plt.xlabel('Logon IDs', fontsize=12)
    plt.ylabel('Hamming Distance', fontsize=12)
    plt.title('The query results of logon behaviors', fontsize=14, fontweight='bold')

    # 构造图例
    legend_patches = []
    for level, colors in color_variations.items():
        for i, color in enumerate(colors):
            label = f"{level} ({overlap_bins[i]}-{overlap_bins[i + 1]}%)"
            legend_patches.append(mpatches.Patch(color=color, label=label))

    plt.legend(handles=legend_patches, fontsize=10, title="Threat Level and Overlap Percentage")

    plt.tight_layout()
    plt.show()




def extract_multiple_data_with_employee_info(results, sample_id, distances, output_csv, employee_csv):
    """
    提取多个 data_id 对应的数据并保存到一个 CSV 文件中，添加汉明距离信息，
    输出数据的排序逻辑为：第一行为样本数据，其余数据按照汉明距离从小到大排序。
    :param results: 查询到的 data ID 列表。
    :param sample_id: 样本数据的 ID。
    :param distances: 对应的汉明距离列表（与 results 对应）。
    :param output_csv: 输出的 CSV 文件名。
    :param employee_csv: 员工信息文件名。
    """
    # 提取文件前缀
    try:
        file_prefix = sample_id.split('_')[0] + '_' + sample_id.split('_')[1]
    except IndexError:
        raise ValueError("无法从 sample_id 中提取文件前缀。")
    
    # 构造文件名
    file_name = f"../dataset/Unstructured_Data/{file_prefix}.csv"
    
    # 提取纯数据 ID
    pure_results = [data_id.split('_', 2)[-1] for data_id in results]
    pure_sample_id = sample_id.split('_', 2)[-1]

    # 读取 CSV 文件
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 '{file_name}' 未找到。")

    # 查找所有 results 对应的行
    extracted_rows = df[df['id'].isin(pure_results)]
    
    if extracted_rows.empty:
        missing_ids = set(pure_results) - set(df['id'])
        raise ValueError(f"在文件 '{file_name}' 中未找到指定的 ID。这些 ID 缺失: {missing_ids}")

    # 读取员工信息文件
    try:
        employee_df = pd.read_csv(employee_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"员工信息文件 '{employee_csv}' 未找到。")

    # 合并 user 列信息
    merged_df = pd.merge(
        extracted_rows, employee_df[['user_id', 'role', 'functional_unit']],
        left_on='user', right_on='user_id', how='left'
    )

    # 如果有未匹配的用户，提示警告
    unmatched_users = merged_df[merged_df['role'].isnull()]['user'].unique()
    if len(unmatched_users) > 0:
        print("警告：以下用户未找到匹配的员工信息:", unmatched_users)

    # 创建汉明距离映射字典
    hamming_distance_dict = dict(zip(pure_results, distances))

    # 添加汉明距离列
    merged_df['Hamming_Distance'] = merged_df['id'].apply(
        lambda x: hamming_distance_dict.get(x, "N/A")  # 获取汉明距离，如果找不到，标记为 "N/A"
    )

    # 标记样本数据
    merged_df['Is_Sample'] = merged_df['id'].apply(lambda x: x == pure_sample_id)

    # 排序逻辑：样本数据优先，其余数据按汉明距离升序排列
    sorted_df = merged_df.sort_values(by=['Is_Sample', 'Hamming_Distance'], ascending=[False, True])

    # 删除辅助列
    sorted_df = sorted_df.drop(columns=['user_id', 'Is_Sample'])

    # 保存到新的 CSV 文件
    sorted_df.to_csv(output_csv, index=False)
    print(f"已提取数据并保存到文件 '{output_csv}'。")
