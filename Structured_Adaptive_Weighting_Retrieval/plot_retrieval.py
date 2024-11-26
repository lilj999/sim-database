import random
import time
import torch
from DataBase.SimHash import SimHashDatabase
from utils.result_process_structured import plot_query_results_with_categories
from data_process.data_process_structured import extract_min_max, process_result_data, process_and_sample_data
from utils.embedding_achieve import UserEmbeddingNet

# Paths
DATA_PATH = '../dataset/Structured_Data/filtered_data.csv'
LABEL_PATH = '../dataset/Structured_Data/filtered_labels.csv'
MIN_MAX_PATH = '../dataset/Structured_Data/min_max_values.csv'
MODEL_PATH = '../model/model_all_data_2024-11-20-17-59.pth'

# Main function
if __name__ == "__main__":
    # Extract min-max data
    min_max_df = extract_min_max(MIN_MAX_PATH)

    # Initialize the model
    input_dim = len(min_max_df)  # Input dimension matches the number of features
    num_classes = 6  # Number of classes
    embedding_dim = 128  # Embedding dimension
    model = UserEmbeddingNet(input_dim, embedding_dim, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # Process all rows
    results = process_and_sample_data(DATA_PATH,LABEL_PATH, model,num_samples_per_label=100)

    sample_data, multiple_data_group = process_result_data(results,sample_category="3.0")

    # Check if sample data exists
    if not sample_data:
        raise ValueError("No sample data found.")
    sample_id, sample_weights = sample_data[0]
    print(f"Sample ID: {sample_id}")
    print(f"Sample Weights: {sample_weights}")

    # Initialize SimHash database
    simhashdatabase = SimHashDatabase(num_bits=64, num_segments=4, hamming_threshold=4)

    # Process multiple data group
    print("\nProcessing Multiple Data Group:")
    for data_id, row_weights in multiple_data_group:
        print(data_id)
        print(row_weights)
        simhashdatabase.add_to_database(data_id, row_weights)

    # Query the sample data
    print("\nQuerying Sample Data:")
    start_time = time.time()
    results, distances = simhashdatabase.query(sample_weights)
    end_time = time.time()

    # Sort results by distance
    sorted_results = sorted(zip(results, distances), key=lambda x: x[1])
    sorted_results, sorted_distances = zip(*sorted_results)

    # Print sorted query results
    print("Sorted Query Results:")
    for result, distance in zip(sorted_results, sorted_distances):
        print(f"Result: {result}, Distance: {distance}")
    print(f"Query Time: {end_time - start_time:.4f} seconds")

    # Plot the query results
    plot_query_results_with_categories(sorted_results, sorted_distances, sample_id, zero_height=0.5)
