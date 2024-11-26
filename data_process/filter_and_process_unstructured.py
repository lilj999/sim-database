

import pandas as pd
from ..DataBase.WeightMapping import WeightMapping
from data_process_unstrucutred import extract_information
from ..DataBase.SimHash import SimHashDatabase
import random
import time

# 初始化数据文件路径
data_file = "filtered_device.csv"  # 数据文件路径
employee_file = "employee_detailed_tenure.csv"  # 员工在职记录文件路径
output_csv = 'select_device.csv'  # 输出筛选结果的 CSV 文件路径

# 提取数据
processed_data, data_ids = extract_information(data_file, employee_file, read_all=False, sample_size=50000)

# 初始化 WeightMapping 和 SimHashDatabase
weight_mapping = WeightMapping()
simhashdatabase = SimHashDatabase(num_bits=64, num_segments=4, hamming_threshold=10)

# 打乱数据顺序以保证随机性
random_indices = list(range(len(processed_data)))
random.shuffle(random_indices)  # 随机打乱索引

# 存储筛选结果
data_with_weights = []

for idx in random_indices:
    # 获取与当前行对应的 data_id
    current_data_id = data_ids[idx]

    # 提取用户 ID 并跳过不在名录中的用户
    current_user = processed_data.iloc[idx]["user"]
    valid_users = set(pd.read_csv(employee_file)["user_id"])  # 加载员工名录
    if current_user not in valid_users:
        continue

    # 提取关键词权重并计算总权重
    row_weights = weight_mapping.extract_keywords_weights(processed_data.iloc[[idx]])[0]
    total_weight = weight_mapping.calculate_total_weight(row_weights)

    # 插入当前行数据到 SimHashDatabase
    simhashdatabase.add_to_database(current_data_id, row_weights)

    # 添加 Total_Weight 列
    row = processed_data.iloc[idx].copy()  # 获取当前行数据
    row["Total_Weight"] = total_weight  # 添加总权重
    data_with_weights.append(row)

# 转换为 DataFrame
df = pd.DataFrame(data_with_weights)

# 筛选数据
low_weight = df[df['Total_Weight'] < 20].sample(n=400, replace=True, random_state=random.randint(0, 10000))  # 权重 < 20
medium_weight = df[(df['Total_Weight'] >= 20) & (df['Total_Weight'] <= 30)].sample(n=200, replace=True, random_state=random.randint(0, 10000))  # 20 <= 权重 <= 30
high_weight = df[df['Total_Weight'] > 30].sample(n=100, replace=True, random_state=random.randint(0, 10000))  # 权重 > 30

# 添加 Threat_Level 标签
low_weight['Threat_Level'] = 'Low'
medium_weight['Threat_Level'] = 'Medium'
high_weight['Threat_Level'] = 'High'

# 合并筛选结果
filtered_data = pd.concat([low_weight, medium_weight, high_weight])

# 删除不需要的列
columns_to_drop = ['activity', 'is_resigned', 'role', 'function_unit', 'Time_Feature']
filtered_data = filtered_data.drop(columns=columns_to_drop)

# 保存到 CSV 文件
filtered_data.to_csv(output_csv, index=False)

# 打印输出信息
print(f"筛选后的数据已保存至 {output_csv}")
