import pandas as pd
from WeightMapping import WeightMapping
from data_process import extract_information
from SimHash import SimHashDatabase
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from result_process import plot_hamming_distances_histogram, extract_multiple_data_with_employee_info

class BaselineSearch:
    def __init__(self, database_weights, database_ids):
        self.database_weights = database_weights
        self.database_ids = database_ids
        self.all_keys = self.get_all_keys()

    def get_all_keys(self):
        """提取所有唯一的关键词，确保向量维度一致"""
        all_keys = set()
        for weights in self.database_weights:
            all_keys.update(weights.keys())
        return sorted(all_keys)

    def convert_to_vector(self, weight):
        """将权重字典转换为固定长度向量"""
        return np.array([weight.get(key, 0) for key in self.all_keys])

    # 暴力搜索
    def brute_force_search(self, query_weight):
        query_vector = self.convert_to_vector(query_weight)
        distances = []
        for idx, db_weight in enumerate(self.database_weights):
            db_vector = self.convert_to_vector(db_weight)
            distance = np.sum(np.abs(query_vector - db_vector))  # 曼哈顿距离
            distances.append((self.database_ids[idx], distance))
        # 按距离升序排序并返回前10个结果
        distances.sort(key=lambda x: x[1])
        results = [item[0] for item in distances[:10]]
        top_distances = [item[1] for item in distances[:10]]
        return results, top_distances

    # 基于余弦相似度的检索
    def cosine_similarity_search(self, query_weight):
        query_vector = self.convert_to_vector(query_weight)
        db_vectors = np.array([self.convert_to_vector(w) for w in self.database_weights])
        similarities = cosine_similarity([query_vector], db_vectors).flatten()
        sorted_indices = np.argsort(-similarities)  # 按相似度降序排序
        results = [self.database_ids[idx] for idx in sorted_indices[:10]]  # 取前10个结果
        top_similarities = [similarities[idx] for idx in sorted_indices[:10]]
        return results, top_similarities

if __name__ == "__main__":
    # 输入文件路径
    data_file = "processed_file.csv"
    employee_file = "employee_detailed_tenure.csv"
    output_csv = '1.csv'

    # 从 extract_information 函数中提取数据
    processed_data, data_ids = extract_information(data_file, employee_file, read_all=False, sample_size=1000)

    # 初始化 WeightMapping 和 SimHashDatabase
    weight_mapping = WeightMapping()
    simhashdatabase = SimHashDatabase(num_bits=64, num_segments=4, hamming_threshold=10)

    # 提取关键词及其权重
    extracted_weights = weight_mapping.extract_keywords_weights(processed_data)

    # 遍历每一行数据，并逐行插入到 SimHashDatabase 中
    for idx, row_weights in enumerate(extracted_weights[0:1000]):
        current_data_id = data_ids[idx]
        simhashdatabase.add_to_database(current_data_id, row_weights)
        print(f"Inserted Row {idx}: ID={current_data_id}, Weights={row_weights}")

    # 初始化基线检索器
    baseline_search = BaselineSearch(extracted_weights, data_ids)

    # 查询示例数据
    sample_data, sample_ids = extract_information(data_file, employee_file, read_all=False, sample_size=1)
    sample_weight = weight_mapping.extract_keywords_weights(sample_data)[0]

    # SimHash检索
    start_time = time.time()
    simhash_results, simhash_distances = simhashdatabase.query(sample_weight)
    simhash_time = time.time() - start_time
    print("SimHash 查询结果：", simhash_results)
    print(f"SimHash 查询耗时：{simhash_time:.6f} 秒")

    # 暴力检索
    start_time = time.time()
    brute_results, brute_distances = baseline_search.brute_force_search(sample_weight)
    brute_time = time.time() - start_time
    print("暴力检索结果：", brute_results)
    print(f"暴力检索耗时：{brute_time:.6f} 秒")

    # 基于余弦相似度的检索
    start_time = time.time()
    cosine_results, cosine_similarities = baseline_search.cosine_similarity_search(sample_weight)
    cosine_time = time.time() - start_time
    print("余弦相似度检索结果：", cosine_results)
    print(f"余弦相似度检索耗时：{cosine_time:.6f} 秒")

    # 可视化哈希距离分布
    plot_hamming_distances_histogram(simhash_results, simhash_distances)

    # 合并 sample_ids 和 results，去重
    merged_ids = list(set(simhash_results).union(set(sample_ids)))
    extract_multiple_data_with_employee_info(data_ids=merged_ids, output_csv=output_csv, employee_csv=employee_file)
