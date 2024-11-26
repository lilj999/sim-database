import matplotlib.pyplot as plt
import pandas as pd

def plot_hamming_distances_histogram(data_ids, distances):
    """
    绘制汉明距离的直方图。
    :param data_ids: 文档 ID 列表。
    :param distances: 对应的汉明距离列表。
    """
    # 将数据按汉明距离从小到大排序
    sorted_data = sorted(zip(data_ids, distances), key=lambda x: x[1])
    sorted_ids, sorted_distances = zip(*sorted_data)
    
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_distances)), sorted_distances, tick_label=sorted_ids)
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel('Document IDs', fontsize=12)
    plt.ylabel('Hamming Distance', fontsize=12)
    plt.title('Hamming Distance Histogram', fontsize=14)
    plt.tight_layout()
    plt.show()


def extract_multiple_data_with_employee_info(data_ids, output_csv, employee_csv):
    """
    提取多个 data_id 对应的数据并保存到一个 CSV 文件中，
    同时从 employee_csv 中根据 user 信息提取 role 和 functional_unit。
    :param data_ids: 数据 ID 列表，例如 ['processed_file_N0K8-P6RL36MD-9480NFAR', ...]。
    :param output_csv: 输出的 CSV 文件名。
    :param employee_csv: 员工信息文件名。
    """
    # 提取文件前缀
    first_data_id = data_ids[0]
    try:
        file_prefix = first_data_id.split('_')[0] + '_' + first_data_id.split('_')[1]
    except IndexError:
        raise ValueError("无法从 data_ids 中提取文件前缀。")
    
    # 构造文件名
    file_name = f"{file_prefix}.csv"
    
    # 提取纯数据 ID
    pure_data_ids = [data_id.split('_', 2)[-1] for data_id in data_ids]

    # 读取 CSV 文件
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 '{file_name}' 未找到。")

    # 查找所有 data_id 对应的行
    extracted_rows = df[df['id'].isin(pure_data_ids)]
    
    if extracted_rows.empty:
        missing_ids = set(pure_data_ids) - set(df['id'])
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

    # 按 user 的首字母排序
    merged_df['user_initial'] = merged_df['user'].str[0]
    sorted_df = merged_df.sort_values(by=['user_initial', 'user'])

    # 删除辅助列
    sorted_df = sorted_df.drop(columns=['user_initial', 'user_id'])

    # 保存到新的 CSV 文件
    sorted_df.to_csv(output_csv, index=False)
    print(f"已提取数据并保存到文件 '{output_csv}'。")