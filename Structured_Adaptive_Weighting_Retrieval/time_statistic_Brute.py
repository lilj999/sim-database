from utils.embedding_achieve import normalize_data, ProposedModel, CATEGORY_WEIGHTS, embedding_generate_with_weights,UserEmbeddingNet
from data_process.data_process_structured import process_and_sample_data_embedding
import torch
import pandas as pd
import numpy as np
import json
import random
from matplotlib import pyplot as plt
import os
import time
import torch
import numpy as np
import pandas as pd
from DataBase.SimHash import SimHashDatabase

# Paths
# DATA_PATH = 'dataset\\filtered_data_1.csv'
# LABEL_PATH = 'dataset\\filtered_labels_1.csv'

DATA_PATH = '../dataset/Structured_Data/filtered_data.csv'
LABEL_PATH = '../dataset/Structured_Data/filtered_labels.csv'
MODEL_PATH = '../model/model_all_data_2024-11-20-17-59.pth'

# Sample sizes to test
SAMPLE_SIZES = [5000, 10000, 15000, 20000, 30000, 50000, 100000,150000]

# CSV file to save results
OUTPUT_CSV = "query_times_embedding.csv"

def main():
    # Initialize the model
    input_dim = 76  # Input dimension matches the number of features in the data
    num_classes = 6  # Number of classes
    embedding_dim = 128  # Embedding dimension
    model = UserEmbeddingNet(input_dim, embedding_dim, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # Initialize a list to store results
    results_list = []

    # Iterate over each sample size
    for sample_size in SAMPLE_SIZES:
        print(f"Testing with {sample_size} samples...")

        # Process and sample data
        samples, results = process_and_sample_data_embedding(DATA_PATH, LABEL_PATH, model, num_samples_per_label=sample_size)

        # Ensure all embeddings are numeric arrays
        cleaned_results = []
        for feature_name, embedding, weight in results:
            try:
                numeric_embedding = np.array(embedding, dtype=np.float64)
                cleaned_results.append((feature_name, numeric_embedding, weight))
            except ValueError as e:
                print(f"Invalid embedding detected and skipped: {embedding}")
        results = cleaned_results

        # Initialize SimHash database
        simhashdatabase = SimHashDatabase(num_bits=64, num_segments=4, hamming_threshold=4)

        # Add data to SimHash database
        for feature_name, embedding, weight in results:
            # Convert embedding to dictionary format with indices as keys
            embedding_dict = {f"feature_{i}": float(value) for i, value in enumerate(embedding)}
            simhashdatabase.add_to_database(feature_name, embedding_dict)

        # Query each sample and measure time
        for idx, sample in enumerate(samples):
            # Convert sample embedding to dictionary format
            query_embedding = {f"feature_{i}": float(value) for i, value in enumerate(sample[1])}

            start_time = time.time()
            simhashdatabase.query(query_embedding)  # Perform search
            end_time = time.time()

            total_search_time = end_time - start_time
            print(f"Sample {idx + 1}: Query Time = {total_search_time:.4f} seconds")

            # Append result to the list
            results_list.append({
                "Sample_Size": sample_size,
                "Sample_Index": idx + 1,
                "Query_Time": total_search_time
            })

    # Save results to a CSV file
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"Query times saved to {OUTPUT_CSV}")
if __name__ == "__main__":
    main()