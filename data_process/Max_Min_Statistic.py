import pandas as pd


'''
计算整个数据集每一列的最大值和最小值，以便于归一化
'''
# 定义函数：计算每列的最大值和最小值
def compute_min_max(input_file, output_file):
    # 读取输入的CSV文件
    data = pd.read_csv(input_file)
    
    # 计算每列的最大值和最小值
    min_values = data.min()
    max_values = data.max()
    
    # 将结果组织成一个DataFrame
    result = pd.DataFrame({
        'Column': data.columns,
        'MinValue': min_values.values,
        'MaxValue': max_values.values
    })
    
    # 保存结果到新的CSV文件
    result.to_csv(output_file, index=False)
    print(f"最大值和最小值已保存到 {output_file}")

# 配置文件路径
input_file_path = '../dataset/filtered_data.csv'  # 输入CSV文件路径
output_file_path = '../dataset/min_max_values.csv'  # 输出CSV文件路径

# 运行程序
if __name__ == "__main__":
    compute_min_max(input_file_path, output_file_path)
