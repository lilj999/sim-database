import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DataBase.WeightMapping import WeightMapping
from data_process import data_process_unstrucutred

from DataBase.SimHash import SimHashDatabase


from sklearn.metrics.pairwise import cosine_similarity
import time


class BaselineSearch:
    def __init__(self, database_weights, database_ids):
        self.database_weights = database_weights
        self.database_ids = database_ids
        self.all_keys = self.get_all_keys()

    def get_all_keys(self):
        all_keys = set()
        for weights in self.database_weights:
            all_keys.update(weights.keys())
        return sorted(all_keys)

    def convert_to_vector(self, weight):
        return np.array([weight.get(key, 0) for key in self.all_keys])

    def brute_force_search(self, query_weight):
        query_vector = self.convert_to_vector(query_weight)
        distances = []
        for idx, db_weight in enumerate(self.database_weights):
            db_vector = self.convert_to_vector(db_weight)
            distance = np.sum(np.abs(query_vector - db_vector))
            distances.append((self.database_ids[idx], distance))
        distances.sort(key=lambda x: x[1])
        return [item[0] for item in distances[:10]], [item[1] for item in distances[:10]]

    def cosine_similarity_search(self, query_weight):
        query_vector = self.convert_to_vector(query_weight)
        db_vectors = np.array([self.convert_to_vector(w) for w in self.database_weights])
        similarities = cosine_similarity([query_vector], db_vectors).flatten()
        sorted_indices = np.argsort(-similarities)
        return [self.database_ids[idx] for idx in sorted_indices[:10]], [similarities[idx] for idx in sorted_indices[:10]]


def measure_search_times(database_weights, database_ids, sample_weight, sizes):
    simhash_times = []
    brute_force_times = []
    cosine_similarity_times = []

    for size in sizes:
        print(f"\nTesting with database size: {size}")
        current_weights = database_weights[:size]
        current_ids = database_ids[:size]

        # Initialize SimHash and BaselineSearch
        simhash_database = SimHashDatabase(num_bits=64, num_segments=4, hamming_threshold=10)
        for idx, weights in enumerate(current_weights):
            simhash_database.add_to_database(current_ids[idx], weights)
        baseline_search = BaselineSearch(current_weights, current_ids)

        # Measure SimHash time and output results
        start_time = time.time()
        simhash_results, simhash_distances = simhash_database.query(sample_weight)
        simhash_time = time.time() - start_time
        simhash_times.append(simhash_time)
        print(f"SimHash Results: {simhash_results}")
        print(f"SimHash Time: {simhash_time:.6f} seconds")

        # Measure Brute Force time and output results
        start_time = time.time()
        brute_results, brute_distances = baseline_search.brute_force_search(sample_weight)
        brute_force_time = time.time() - start_time
        brute_force_times.append(brute_force_time)
        print(f"Brute Force Results: {brute_results}")
        print(f"Brute Force Time: {brute_force_time:.6f} seconds")

        # Measure Cosine Similarity time and output results
        start_time = time.time()
        cosine_results, cosine_similarities = baseline_search.cosine_similarity_search(sample_weight)
        cosine_similarity_time = time.time() - start_time
        cosine_similarity_times.append(cosine_similarity_time)
        print(f"Cosine Similarity Results: {cosine_results}")
        print(f"Cosine Similarity Time: {cosine_similarity_time:.6f} seconds")

    return simhash_times, brute_force_times, cosine_similarity_times


if __name__ == "__main__":
    data_file = "../dataset/Unstructured_Data/processed_file.csv"
    employee_file = "../dataset/Unstructured_Data/employee_detailed_tenure.csv"

    # Extract data
    processed_data, data_ids = data_process_unstrucutred.extract_information(data_file, employee_file, read_all=False, sample_size=10000)
    weight_mapping = WeightMapping()
    extracted_weights = weight_mapping.extract_keywords_weights(processed_data)

    # Prepare sample query
    # sample_data, _ = extract_information(data_file, employee_file, read_all=False, sample_size=1)
    sample_data, _ =data_process_unstrucutred.extract_sample_nformation(data_file, employee_file, read_all=False, sample_size=1)
    sample_weight = weight_mapping.extract_keywords_weights(sample_data)[0]

    # Database sizes to test
    database_sizes = [100, 300, 1000, 3000, 5000, 7000, 10000]  # 更合理的规模设置

    # Measure times
    simhash_times, brute_force_times, cosine_similarity_times = measure_search_times(
        extracted_weights, data_ids, sample_weight, database_sizes
    )

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(database_sizes, simhash_times, label="The proposed method", marker="o")
    plt.plot(database_sizes, brute_force_times, label="Brute Force", marker="s")
    plt.plot(database_sizes, cosine_similarity_times, label="Cosine Similarity", marker="^")
    plt.xlabel("Number of retrievals")
    plt.ylabel("Average Query Time (seconds)")
    plt.title("Comparison of Three Different Retrieval Methods")
    plt.legend()
    plt.grid()
    plt.show()
