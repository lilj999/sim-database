import time
import torch
import pandas as pd
from SimHash import SimHashDatabase
from data_process import process_and_sample_data, process_result_data
from embedding_achieve import UserEmbeddingNet

# Paths
DATA_PATH = 'dataset\\filtered_data_1.csv'
LABEL_PATH = 'dataset\\filtered_labels_1.csv'
MODEL_PATH = 'model\model_all_data_2024-11-20-17-59.pth'

# Sample sizes to test
SAMPLE_SIZES = [5000, 10000, 15000, 20000, 30000, 50000, 100000, 150000]

# CSV file to save results
OUTPUT_CSV = "query_times_alter2.csv"

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

    # Test for each sample size
    for sample_size in SAMPLE_SIZES:
        print(f"Testing with {sample_size} samples...")
        
        # Process and sample data
        results = process_and_sample_data(DATA_PATH, LABEL_PATH, model, num_samples_per_label=sample_size)
        
        # Get sample data and multiple data group
        sample_data, multiple_data_group = process_result_data(results, sample_category="4.0")

        # Check if sample data exists
        if not sample_data:
            raise ValueError("No sample data found.")
        sample_id, sample_weights = sample_data[0]

        # Initialize SimHash database
        simhashdatabase = SimHashDatabase(num_bits=64, num_segments=4, hamming_threshold=4)

        # Add data to SimHash database
        for data_id, row_weights in multiple_data_group:
            simhashdatabase.add_to_database(data_id, row_weights)

        # Query the sample data and measure time
        start_time = time.time()
        simhashdatabase.query(sample_weights)
        end_time = time.time()

        # Calculate query time
        query_time = end_time - start_time
        print(f"Sample size: {sample_size}, Query time: {query_time:.4f} seconds")

        # Append result to list
        results_list.append({"Sample_Size": sample_size, "Query_Time": query_time})

    # Save results to CSV
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()