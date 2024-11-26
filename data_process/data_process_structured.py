import torch
import pandas as pd
import numpy as np
import random
from utils.embedding_achieve import embedding_generate_with_weights, UserEmbeddingNet
from sklearn.preprocessing import MinMaxScaler

# Extract Min-Max Values
def extract_min_max(csv_path):
    """
    Extract Min and Max values from a CSV file and return as a DataFrame.
    :param csv_path: Path to the CSV file with columns 'MinValue' and 'MaxValue'.
    :return: DataFrame with MinValue and MaxValue.
    """
    data = pd.read_csv(csv_path)
    return pd.DataFrame({
        'MinValue': data['MinValue'].values,
        'MaxValue': data['MaxValue'].values
    })




def process_and_sample_data(data_file, label_file, model, num_samples_per_label=100):
    """
    Process raw data with corresponding labels, normalize, sample by label, and generate embeddings.
    
    :param data_file: Path to the input data file (CSV format).
    :param label_file: Path to the label file (CSV format with a 'Label' column).
    :param model: Trained PyTorch model for generating embeddings.
    :param num_samples_per_label: Number of random samples to select per label.
    
    :return: Processed data list with format [(data_id, embedding, weight), ...].
    """
    def normalize_data_with_scaler(data):
        """
        Normalize data using MinMaxScaler from sklearn.
        :param data: Pandas DataFrame containing numerical data.
        :return: Normalized DataFrame.
        """
        scaler = MinMaxScaler()
        normalized_array = scaler.fit_transform(data)
        return pd.DataFrame(normalized_array, columns=data.columns)

    def sample_by_label(data, num_samples_per_label):
        """
        Randomly sample data by labels.
        :param data: DataFrame containing data and a 'Label' column.
        :param num_samples_per_label: Number of samples per label.
        :return: Sampled DataFrame.
        """
        sampled_data = []
        grouped = data.groupby('Label')

        for label, group in grouped:
            sampled_group = group.sample(n=min(num_samples_per_label, len(group)))
            sampled_data.append(sampled_group)

        return pd.concat(sampled_data).reset_index(drop=True)

    def process_all_rows(data):
        """
        Process each row to generate embeddings and weights.
        :param data: DataFrame containing normalized data and 'Label'.
        :return: List of processed data in the format [(data_id, embedding, weight), ...].
        """
        all_data = []
        label_counter = {}

        for _, row in data.iterrows():
            # Extract features and label
            features = row.iloc[:-1].values.astype(float).reshape(1, -1)  # Exclude 'Label' column
            label = row['Label']

            # Generate unique ID for the sample
            if label not in label_counter:
                label_counter[label] = 0
            label_counter[label] += 1
            feature_name = f"{label}-{label_counter[label]}"

            # Generate embeddings and weights using the model
            embeddings, weights = embedding_generate_with_weights(model, features)
            embedding_vector = embeddings.flatten().tolist()
            weight = float(weights.mean())

            # Append processed data
            all_data.append((feature_name, embedding_vector, weight))

        return all_data

    # Load data and labels
    data = pd.read_csv(data_file)
    labels = pd.read_csv(label_file)

    # Ensure the labels match the data length
    if len(data) != len(labels):
        raise ValueError("Data and labels must have the same number of rows.")

    # Merge data and labels
    data['Label'] = labels['Label']  # Assumes the label column is named 'Label'

    # Normalize data (excluding the 'Label' column)
    features = data.drop(columns=['Label'])
    normalized_data = normalize_data_with_scaler(features)
    normalized_data['Label'] = data['Label']  # Add the label column back

    # Sample data by label
    sampled_data = sample_by_label(normalized_data, num_samples_per_label)

    # Process sampled rows
    processed_data = process_all_rows(sampled_data)

    return processed_data

def process_and_sample_data_embedding(data_path, label_path, model, num_samples_per_label=10):
    # Load data and labels
    data = pd.read_csv(data_path)
    labels = pd.read_csv(label_path)

    # Ensure the labels match the data length
    if len(data) != len(labels):
        raise ValueError("Data and labels must have the same number of rows.")

    # Merge data and labels
    data['Label'] = labels['Label']

    # Normalize data (excluding the 'Label' column)
    features = data.drop(columns=['Label'])
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(features)

    # Combine normalized features and labels
    normalized_data = pd.DataFrame(normalized_data, columns=features.columns)
    normalized_data['Label'] = data['Label']

    # Sample data by label
    sampled_data = []
    grouped = normalized_data.groupby('Label')
    for label, group in grouped:
        sampled_group = group.sample(n=min(num_samples_per_label, len(group)))
        sampled_data.append(sampled_group)

    sampled_data = pd.concat(sampled_data).reset_index(drop=True)

    # Generate embeddings
    samples, nodes = [], []
    for idx, row in sampled_data.iterrows():
        features = row.iloc[:-1].values.astype(float).reshape(1, -1)
        label = row['Label']

        embeddings, weights = embedding_generate_with_weights(model, features)

        # Ensure embeddings are numeric
        embeddings = np.array(embeddings, dtype=np.float64)

        feature_name = f"{label}-{idx + 1}"
        embedding_vector = embeddings.flatten().tolist()
        weight = float(weights.mean())

        if idx == 0:  # Take the first row as sample
            samples.append((feature_name, embedding_vector, weight))
        else:
            nodes.append((feature_name, embedding_vector, weight))

    return samples, nodes

# Normalize Features
def normalize_features(data_list):
    """
    Normalize each column of the dataset.
    :param data_list: List of embeddings.
    :return: Normalized embeddings.
    """
    if not data_list:
        raise ValueError("The input data_list is empty. Cannot normalize features.")
    data_matrix = np.array(data_list)
    if data_matrix.size == 0:
        raise ValueError("The data matrix is empty after conversion. Cannot normalize features.")
    min_vals = data_matrix.min(axis=0)
    max_vals = data_matrix.max(axis=0)
    range_vals = max_vals - min_vals + 1e-15
    return ((data_matrix - min_vals) / range_vals).tolist()

# Process Result Data
def process_result_data(result, sample_category=None, sample_count=1):
    """
    Split data into samples and other data, and normalize embeddings.
    :param result: List of tuples in the format (data_id, embedding).
    :param sample_category: Specify which category to sample from (e.g., "5.0").
    :param sample_count: Number of random samples to extract from the specified category.
    :return: Two lists: samples and other_data.
    """
    if not result:
        raise ValueError("The input result data is empty. Please provide valid data.")

    # 存储所有嵌入向量和 ID
    all_embeddings, all_ids = [], []

    # 提取有效数据
    for entry in result:
        if isinstance(entry, tuple) and len(entry) >= 2:
            data_id, embedding = entry[0], entry[1]
            if isinstance(embedding, list) and all(isinstance(e, (int, float)) for e in embedding):
                all_ids.append(data_id)
                all_embeddings.append(embedding)

    if not all_embeddings:
        raise ValueError("No valid embeddings found in the input result data.")

    # 对所有数据的每一列进行归一化
    data_matrix = np.array(all_embeddings)
    min_vals, max_vals = data_matrix.min(axis=0), data_matrix.max(axis=0)
    range_vals = max_vals - min_vals + 1e-15
    normalized_matrix = (data_matrix - min_vals) / range_vals

    # 将归一化数据与 ID 结合，并构造输出格式为 (data_id, embedding)
    all_data = [
        (data_id, {f"feature_{i}": normalized_matrix[idx][i] for i in range(len(normalized_matrix[0]))})
        for idx, data_id in enumerate(all_ids)
    ]

    # 筛选出指定类别的数据
    if sample_category:
        filtered_data = [
            entry for entry in all_data if entry[0].startswith(sample_category)
        ]
        if not filtered_data:
            raise ValueError(f"No data found for the specified category: {sample_category}")
    else:
        filtered_data = all_data

    # 随机选取 samples
    sample_indices = set(random.sample(range(len(filtered_data)), min(sample_count, len(filtered_data))))
    samples = [filtered_data[idx] for idx in sample_indices]

    # 剩余数据作为 other_data
    other_data = [entry for entry in all_data if entry not in samples]

    return samples, other_data