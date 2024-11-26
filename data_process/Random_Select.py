import pandas as pd
import random

# 定义函数：从文件中加载数据
def load_csv(file_path):
    return pd.read_csv(file_path)

# 定义函数：根据标签随机选择数据
def sample_by_label(data, labels, num_samples=100):
    # 合并数据和标签
    data['Label'] = labels
    sampled_data = []

    # 按标签进行分组
    grouped = data.groupby('Label')
    for label, group in grouped:
        # 随机选择样本
        sampled_group = group.sample(n=min(num_samples, len(group)), random_state=42)
        sampled_data.append(sampled_group)

    # 合并所有分组后的数据
    result = pd.concat(sampled_data).reset_index(drop=True)
    return result

# 定义主函数
def main(data_file, label_file, output_file):
    # 读取数据文件和标签文件
    data = load_csv(data_file)
    labels = load_csv(label_file)['Label']  # 假设标签文件中列名为 'Label'

    # 检查数据行数和标签行数是否匹配
    if len(data) != len(labels):
        raise ValueError("数据和标签的行数不一致！")

    # 对数据按标签随机抽样
    sampled_data = sample_by_label(data, labels)

    # 添加编号列
    sampled_data.insert(0, 'ID', range(1, len(sampled_data) + 1))

    # 保存到新的 CSV 文件
    sampled_data.to_csv(output_file, index=False)
    print(f"随机抽样后的数据已保存到 {output_file}")

# 配置文件路径
data_file_path = '../dataset/filtered_data.csv'  # 待筛选数据文件路径
label_file_path = '../dataset/filtered_labels.csv'  # 标签文件路径

output_file_path = '../dataset/sampled_data.csv'  # 输出文件路径

# 运行程序
if __name__ == "__main__":
    main(data_file_path, label_file_path, output_file_path)
