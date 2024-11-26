import pandas as pd
from DataBase.WeightMapping import WeightMapping
from data_process.data_process_unstrucutred import extract_information,extract_sample_nformation
from DataBase.SimHash import SimHashDatabase
import time  # 用于测量时间
from utils.result_process_unstructured import plot_hamming_distances_histogram, extract_multiple_data_with_employee_info
if __name__ == "__main__":
    import time

    # 输入文件路径
    data_file = "../dataset/Unstructured_Data/select_logon.csv"  # 数据文件路径
    # employee_file = "employee_detailed_tenure.csv"  # 员工在职记录文件路径
    employee_file = "../dataset/Unstructured_Data/employee_detailed_tenure.csv"  # 员工在职记录文件路径
    output_csv = '1.csv'  # 检索出的数据（方便检查）

    # 提取数据
    processed_data, data_ids = extract_information(data_file, employee_file, read_all=True, sample_size=1000)

    # 初始化 WeightMapping 和 SimHashDatabase
    weight_mapping = WeightMapping()
    simhashdatabase = SimHashDatabase(num_bits=64, num_segments=4, hamming_threshold=10)

    # 提取关键词及其权重
    extracted_weights = weight_mapping.extract_keywords_weights(processed_data)

    for idx, row_weights in enumerate(extracted_weights):
        # 获取当前行的 data_id
        current_data_id = data_ids[idx]

        # 计算总权重
        total_weight = weight_mapping.calculate_total_weight(row_weights)

        # 插入到 SimHashDatabase
        simhashdatabase.add_to_database(current_data_id, row_weights)

        print(f"Inserted Row {idx}: ID={current_data_id}, Weights={row_weights}, Total Weight={total_weight}")

    # 查询示例数据
    sample_data, sample_ids = extract_sample_nformation(data_file, employee_file, threat_level='High', sample_size=1)
    sample_weight = weight_mapping.extract_keywords_weights(sample_data)

    if isinstance(sample_weight, list) and len(sample_weight) > 0:
        sample_weight = sample_weight[0]

    # 测量查询时间
    start_time = time.time()
    results, distances = simhashdatabase.query(sample_weight)
    end_time = time.time()

    print("查询结果：", results)
    print(f"查询耗时：{end_time - start_time:.4f} 秒")
    print("样本 ID：", sample_ids)

    # 提取并保存数据
    extract_multiple_data_with_employee_info(
        results=results,
        sample_id=sample_ids[0],
        distances=distances,
        output_csv=output_csv,
        employee_csv=employee_file
    )

    # 读取保存的输出数据以提取 Threat_Level 和权重信息
    import pandas as pd
    output_data = pd.read_csv(output_csv)

    # Threat_Level 和权重信息
    threat_levels = output_data['Threat_Level'].tolist()
    retrieved_weights = weight_mapping.extract_keywords_weights(output_data)

    # 计算权重映射重合度
    overlap_percentages = []
    for r_weight in retrieved_weights:
        overlap_count = len(set(r_weight.keys()) & set(sample_weight.keys()))
        total_count = len(set(r_weight.keys()) | set(sample_weight.keys()))
        overlap_percentages.append((overlap_count / total_count) * 100 if total_count > 0 else 0)

    # 绘制汉明距离直方图和权重映射分布
    plot_hamming_distances_histogram(
        data_ids=output_data['id'].tolist(),
        distances=output_data['Hamming_Distance'].tolist(),
        threat_levels=threat_levels,
        sample_weights=sample_weight,
        retrieved_weights=retrieved_weights
    )
