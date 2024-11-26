import pandas as pd

# 读取数据文件和标签文件
# 读取数据文件和标签文件

'''
过滤数据量较少的数据和ddos数据
类别映射:
原类别: 0 -> 新类别: 0
原类别: 4 -> 新类别: 1
原类别: 5 -> 新类别: 2
原类别: 6 -> 新类别: 3
原类别: 7 -> 新类别: 4
原类别: 8 -> 新类别: 5
'''
data_file = '../dataset/Data.csv'  # 数据文件
label_file = '../dataset/Label.csv'  # 标签文件

data_df = pd.read_csv(data_file)
labels_df = pd.read_csv(label_file)

# 删除 id=1 的数据

filtered_labels_df = labels_df[~labels_df['Label'].isin([1, 2, 3, 9])].reset_index(drop=True)
filtered_data_df = data_df[~labels_df['Label'].isin([1, 2, 3, 9])].reset_index(drop=True)
# 重新对类别进行编号
unique_labels = sorted(filtered_labels_df['Label'].unique())
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

filtered_labels_df['Label'] = filtered_labels_df['Label'].map(label_mapping)

# 打印原始类别到新类别的映射
print("类别映射:")
for old_label, new_label in label_mapping.items():
    print(f"原类别: {old_label} -> 新类别: {new_label}")

# 更新数据文件和标签文件并保存
filtered_labels_df = filtered_labels_df[['Label']]  # 只保留新的类别
filtered_data_df.reset_index(drop=True, inplace=True)
filtered_labels_df.reset_index(drop=True, inplace=True)

filter_data_path='../dataset/filtered_data.csv'
filter_label_path='../dataset/filtered_labels.csv'
filtered_data_df.to_csv(filter_data_path, index=False)
filtered_labels_df.to_csv(filter_label_path, index=False)

print(f"\n过滤后的数据已保存为{filter_data_path}和 {filter_label_path}")
