import pandas as pd
import os
from glob import glob
from collections import Counter
import re

def count_daily_operations(input_file, count_col='tenure'):
    """
    统计CSV文件中 `daily_operation_count` 列中各个类别的个数。

    参数:
    input_file (str): 输入CSV文件路径
    count_col (str): 要统计的列名

    返回:
    None: 打印统计结果
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 检查是否包含目标列
    if count_col not in df.columns:
        print(f"列 '{count_col}' 不存在，请检查文件。")
        return

    # 统计各个类别的个数
    counts = df[count_col].value_counts()

    # 打印统计结果
    print(f"'{count_col}' 列中各个类别的个数：")
    print(counts)

def process_daily_operations(input_file, output_file, date_col='date', user_col='user'):
    """
    从CSV文件中读取数据，为每位用户在特定日期的操作编号，并将结果保存到新的CSV文件中。
    
    参数:
    input_file (str): 输入CSV文件路径
    output_file (str): 输出CSV文件路径
    date_col (str): 日期列的列名
    user_col (str): 用户列的列名
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 将日期列转为日期时间格式
    df[date_col] = pd.to_datetime(df[date_col])

    # 提取日期（不改变原日期时间格式）用于分组
    df['operation_date'] = df[date_col].dt.date

    # 为每位用户在每天的操作编号
    df['daily_operation_count'] = df.groupby([user_col, 'operation_date']).cumcount() + 1

    # 删除临时列，确保输出的内容中日期列保持原来的时间信息
    df = df.drop(columns=['operation_date'])

    # 保存结果到新的CSV文件
    df.to_csv(output_file, index=False)

    print(f"处理完成，结果已保存到: {output_file}")


def count_daily_operations(input_file, count_col='role'):
    """
    统计CSV文件中 `daily_operation_count` 列中各个类别的个数。

    参数:
    input_file (str): 输入CSV文件路径
    count_col (str): 要统计的列名

    返回:
    None: 打印统计结果
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 检查是否包含目标列
    if count_col not in df.columns:
        print(f"列 '{count_col}' 不存在，请检查文件。")
        return

    # 统计各个类别的个数
    counts = df[count_col].value_counts()

    # 打印统计结果
    print(f"'{count_col}' 列中各个类别的个数：")
    print(counts)

def calculate_tenure_v2(file_folder, output_file):
    """
    根据文件夹中的CSV文件计算每个员工的详细任职时间。
    如果员工的身份信息（除 supervisor 列外）完全一致，视为同一员工记录。

    参数:
    file_folder (str): 包含CSV文件的文件夹路径
    output_file (str): 输出文件路径
    """
    # 获取所有CSV文件路径
    csv_files = sorted(glob(os.path.join(file_folder, "*.csv")))

    # 创建一个空DataFrame存储所有数据
    all_data = []

    # 解析每个CSV文件
    for file in csv_files:
        # 获取文件名中的年月信息
        month_year = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file)
        df['month_year'] = month_year  # 添加文件的年月信息到DataFrame中
        all_data.append(df)

    # 合并所有文件的数据
    all_data = pd.concat(all_data, ignore_index=True)

    # 去除 supervisor 列，作为区分员工身份的依据
    identity_columns = [col for col in all_data.columns if col not in ['supervisor', 'month_year']]
    
    # 确保对关键字段排序以便分组
    all_data = all_data.sort_values(by=identity_columns + ['month_year'])

    # 计算任职时间
    result = []
    for group_key, group in all_data.groupby(identity_columns):
        group = group.sort_values(by='month_year')
        start_month = group['month_year'].iloc[0]
        end_month = group['month_year'].iloc[-1]
        
        # 对于同一员工，将所有记录合并并添加任职时间
        for _, row in group.iterrows():
            result.append({
                **row.to_dict(),
                "tenure": f"{start_month}-{end_month}"
            })

    # 去重并保留最后的任职时间
    result_df = pd.DataFrame(result).drop_duplicates(subset=identity_columns, keep='last')

    # 保存结果到CSV文件
    result_df.to_csv(output_file, index=False)
    print(f"详细任职时间结果已保存到: {output_file}")

def filter_csv_by_date(input_file, output_file, start_date, end_date):
    """
    筛选CSV文件中指定日期范围的数据并保存到新的文件。

    参数:
        input_file (str): 输入CSV文件路径。
        output_file (str): 输出CSV文件路径。
        start_date (str): 开始日期（格式: 'YYYY-MM-DD'）。
        end_date (str): 结束日期（格式: 'YYYY-MM-DD'）。
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 将 'date' 列转换为日期时间格式，自动推断格式
    df['date'] = pd.to_datetime(df['date'], dayfirst=False, errors='coerce')  # 设置 dayfirst=False，因为格式是 月/日/年
    
    # 删除无法解析的日期行
    df = df.dropna(subset=['date'])
    
    # 筛选指定日期范围的数据
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    # 保存筛选后的数据到新的CSV文件
    filtered_df.to_csv(output_file, index=False)
    
    print(f"过滤后的数据已保存到: {output_file}")

def check_sensitive_keywords(file_path, column_name='content', keywords=None):
    """
    检查CSV文件中指定列是否包含敏感关键词，并统计其出现频次。
    
    参数:
        file_path (str): CSV文件路径。
        column_name (str): 包含内容的列名，默认是 'content'。
        keywords (list): 需要检查的敏感关键词列表。
    
    返回:
        dict: 每个关键词及其出现次数的字典。
    """
    if keywords is None:
        keywords = [
            "confidential", "proprietary", "strategy", "finance", 
            "contract", "client", "project", "design", 
            "research", "patent"
        ]
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 初始化计数器
        keyword_counts = Counter()

        # 遍历每行内容，统计关键词出现频次
        for content in df[column_name].dropna():
            # 转换为小写，清理标点符号
            words = re.findall(r'\b\w+\b', content.lower())
            # 统计关键词
            for keyword in keywords:
                keyword_counts[keyword] += words.count(keyword)

        return dict(keyword_counts)
    except Exception as e:
        print(f"出现错误: {e}")
        return {}

def extract_information(data_file, employee_file, read_all=True, sample_size=1):
    """
    读取数据文件和员工在职记录，并提取所需输入信息，添加时间特征。
    同时为每行生成唯一数据代号。
    
    参数:
        data_file (str): 数据文件路径（CSV）。
        employee_file (str): 员工在职记录文件路径（CSV）。
        read_all (bool): 如果为 True，读取文件中全部内容；如果为 False，则随机抽取指定数量的数据。
        sample_size (int): 随机抽取的数据行数（仅在 read_all 为 False 时生效）。
    
    返回:
        tuple: 包含以下两部分：
            - DataFrame: 提取后的数据，包括 `is_resigned`, `role`, `function_unit` 和 `Time_Feature`。
            - list: 每行对应的唯一数据代号列表，格式为 `文件名_当前行ID`。
    """
    # 读取数据文件和员工记录文件
    data_df = pd.read_csv(data_file)
    employee_df = pd.read_csv(employee_file)

    # 提取文件名（不含扩展名）
    file_name = os.path.splitext(os.path.basename(data_file))[0]

    # 根据参数决定读取所有行或随机抽取行
    if not read_all:
        if sample_size > len(data_df):
            sample_size = len(data_df)  # 如果样本数大于数据量，取全部数据
        data_df = data_df.sample(n=sample_size)  # 随机抽样

    # 如果缺少 activity 列，添加默认值 "search"
    if "activity" not in data_df.columns:
        data_df["activity"] = "search"

    # 填充其他缺失列为空字符串
    required_columns = ["id", "date", "user", "pc"]
    for col in required_columns:
        if col not in data_df.columns:
            data_df[col] = ""

    # 提取用户信息并匹配员工记录
    def get_employee_info(user, date):
        employee = employee_df[employee_df["user_id"] == user]
        if not employee.empty:
            # 检查是否在职
            tenure = employee.iloc[0]["tenure"]
            try:
                # 解析 tenure，确保只有两个日期值
                tenure_dates = tenure.split("-")
                start_date = tenure_dates[0].strip()
                end_date = tenure_dates[1].strip() if len(tenure_dates) > 1 else ""
                is_resigned = not (start_date <= date <= end_date)
            except IndexError:
                # 如果 tenure 格式异常，默认为离职
                is_resigned = True
            
            # 提取角色和功能单位
            role = employee.iloc[0]["role"]
            function_unit = employee.iloc[0]["functional_unit"]
            return is_resigned, role, function_unit
        else:
            # 如果用户未找到
            return True, "Unknown", "Unknown"

    # 判断时间特征
    def get_time_feature(date):
        try:
            timestamp = pd.to_datetime(date)  # 转换为时间戳
            hour = timestamp.hour  # 提取小时
            if 9 <= hour < 18:
                return "Work"
            else:
                return "Off_Work"
        except Exception:
            return "Unknown"

    # 对每一行数据进行匹配
    extracted_info = data_df.apply(
        lambda row: pd.Series(get_employee_info(row["user"], row["date"])),
        axis=1
    )
    
    # 添加提取的列到数据框
    extracted_info.columns = ["is_resigned", "role", "function_unit"]
    data_df[["is_resigned", "role", "function_unit"]] = extracted_info

    # 添加时间特征列
    data_df["Time_Feature"] = data_df["date"].apply(get_time_feature)

    # 为每一行生成对应的 data_code
    data_code_list = data_df["id"].apply(lambda x: f"{file_name}_{x}").tolist()

    return data_df, data_code_list

def extract_sample_nformation(data_file, employee_file, threat_level=None, read_all=True, sample_size=1):
    """
    读取数据文件和员工在职记录，并提取所需输入信息，添加时间特征。
    同时支持按 Threat_Level 筛选数据并限制返回样本数量。

    参数:
        data_file (str): 数据文件路径（CSV）。
        employee_file (str): 员工在职记录文件路径（CSV）。
        threat_level (str): 要筛选的 Threat_Level（"Low", "Medium", "High"）。
        read_all (bool): 如果为 True，读取文件中全部内容；如果为 False，则随机抽取指定数量的数据。
        sample_size (int): 随机抽取的数据行数（仅在 read_all 为 False 时生效）。

    返回:
        tuple: 包含以下两部分：
            - DataFrame: 提取后的数据，包括筛选结果。
            - list: 每行对应的唯一数据代号列表，格式为 `文件名_当前行ID`。
    """
    # 读取数据文件和员工记录文件
    data_df = pd.read_csv(data_file)
    employee_df = pd.read_csv(employee_file)

    # 提取文件名（不含扩展名）
    file_name = os.path.splitext(os.path.basename(data_file))[0]

    # 如果缺少 activity 列，添加默认值 "search"
    if "activity" not in data_df.columns:
        data_df["activity"] = "search"

    # 填充其他缺失列为空字符串
    required_columns = ["id", "date", "user", "pc"]
    for col in required_columns:
        if col not in data_df.columns:
            data_df[col] = ""

    # 提取用户信息并匹配员工记录
    def get_employee_info(user, date):
        employee = employee_df[employee_df["user_id"] == user]
        if not employee.empty:
            # 检查是否在职
            tenure = employee.iloc[0]["tenure"]
            try:
                tenure_dates = tenure.split("-")
                start_date = tenure_dates[0].strip()
                end_date = tenure_dates[1].strip() if len(tenure_dates) > 1 else ""
                is_resigned = not (start_date <= date <= end_date)
            except IndexError:
                is_resigned = True
            
            # 提取角色和功能单位
            role = employee.iloc[0]["role"]
            function_unit = employee.iloc[0]["functional_unit"]
            return is_resigned, role, function_unit
        else:
            return True, "Unknown", "Unknown"

    # 判断时间特征
    def get_time_feature(date):
        try:
            timestamp = pd.to_datetime(date)
            hour = timestamp.hour
            if 9 <= hour < 18:
                return "Work"
            else:
                return "Off_Work"
        except Exception:
            return "Unknown"

    # 对每一行数据进行匹配
    extracted_info = data_df.apply(
        lambda row: pd.Series(get_employee_info(row["user"], row["date"])),
        axis=1
    )
    extracted_info.columns = ["is_resigned", "role", "function_unit"]
    data_df[["is_resigned", "role", "function_unit"]] = extracted_info

    # 添加时间特征列
    data_df["Time_Feature"] = data_df["date"].apply(get_time_feature)

    # 为每一行生成对应的 data_code
    data_code_list = data_df["id"].apply(lambda x: f"{file_name}_{x}").tolist()

    # 如果 threat_level 参数提供，筛选指定 Threat_Level 的数据
    if threat_level:
        if "Threat_Level" not in data_df.columns:
            raise ValueError("The input data does not contain a 'Threat_Level' column.")

        # 筛选符合 Threat_Level 的数据
        filtered_df = data_df[data_df["Threat_Level"] == threat_level]
        if filtered_df.empty:
            raise ValueError(f"No data found with Threat_Level = {threat_level}")

        # 限制为 sample_size 行
        filtered_df = filtered_df.sample(n=sample_size, replace=False)

        # 更新 data_code_list
        data_code_list = filtered_df["id"].apply(lambda x: f"{file_name}_{x}").tolist()

        return filtered_df, data_code_list

    # 如果未提供 threat_level 参数，根据 read_all 或 sample_size 控制数据量
    if not read_all:
        if sample_size > len(data_df):
            sample_size = len(data_df)
        data_df = data_df.sample(n=sample_size)

    return data_df, data_code_list


def save_to_csv(dataframe, output_file):
    """
    将 DataFrame 保存为 CSV 文件。
    
    参数:
        dataframe (DataFrame): 需要保存的数据。
        output_file (str): 输出文件路径（CSV 文件）。
    """
    dataframe.to_csv(output_file, index=False)
    print(f"数据已保存为 CSV 文件: {output_file}")

# 示例调用
if __name__ == "__main__":
    data_file = "processed_file.csv"  # 数据文件路径
    employee_file = "employee_detailed_tenure.csv"  # 员工在职记录文件路径
    output_file = "processed_file.csv"
    output_csv_file = 'csv.csv'
    count_daily_operations(employee_file)