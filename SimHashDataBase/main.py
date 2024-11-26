import pandas as pd
from WeightMapping import WeightMapping
from data_process import extract_information
from SimHash import SimHashDatabase
import time  # 用于测量时间
from result_process import plot_hamming_distances_histogram, extract_multiple_data_with_employee_info
if __name__ == "__main__":
    # 输入文件路径
    data_file = "processed_file.csv"  # 数据文件路径
    employee_file = "employee_detailed_tenure.csv"  # 员工在职记录文件路径
    output_csv = '1.csv' #检索出的数据（方便检查）
    # 从 extract_information 函数中提取数据
    processed_data, data_ids = extract_information(data_file, employee_file, read_all=False, sample_size=1000)
    #print("Processed Data:")
    #print(processed_data)
    #print("Data IDs:")
    #print(data_ids)

    # 初始化 WeightMapping 和 SimHashDatabase
    weight_mapping = WeightMapping()
    simhashdatabase = SimHashDatabase(num_bits=64,num_segments=4, hamming_threshold=10)

    # 提取关键词及其权重
    extracted_weights = weight_mapping.extract_keywords_weights(processed_data)
    #print("Extracted Weights:")
    #print(extracted_weights)

    # 遍历每一行数据，并逐行插入到 SimHashDatabase 中
    for idx, row_weights in enumerate(extracted_weights[0:100]):
        # 获取与当前行对应的 data_id
        current_data_id = data_ids[idx]

        # 计算当前行的总权重
        total_weight = weight_mapping.calculate_total_weight(row_weights)

        # 插入当前行数据到 SimHashDatabase
        simhashdatabase.add_to_database(current_data_id, row_weights)

        # 打印插入信息
        print(f"Inserted Row {idx}: ID={current_data_id}, Weights={row_weights}, Total Weight={total_weight}")

    # 查询示例数据
    sample_data, sample_ids = extract_information(data_file, employee_file, read_all=False, sample_size=1)
    print(sample_ids)
    sample_weight = weight_mapping.extract_keywords_weights(sample_data)

    if isinstance(sample_weight, list) and len(sample_weight) > 0:
        sample_weight = sample_weight[0]
    print(sample_weight)
    # 测量查询时间
    start_time = time.time()  # 开始时间
    results,distance = simhashdatabase.query(sample_weight)  # 修改为查询单个样本 ID
    end_time = time.time()  # 结束时间

    # 打印查询结果和时间
    print("查询结果：", results)
    print(f"查询耗时：{end_time - start_time:.6f} 秒")
    
    plot_hamming_distances_histogram(results,distance)
    
    # 合并 sample_ids 和 results，去重
    merged_ids = list(set(results).union(set(sample_ids)))

    extract_multiple_data_with_employee_info(data_ids=merged_ids,output_csv=output_csv,employee_csv=employee_file)
