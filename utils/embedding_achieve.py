import torch
import torch.nn as nn
import pandas as pd
import numpy as np
# 数据归一化函数
def extract_min_max(csv_path):
    """
    从CSV文件中提取最小值和最大值，并输出符合指定格式的DataFrame。

    参数：
    - csv_path: CSV文件路径，包含列名 'Column', 'MinValue', 'MaxValue'。

    返回：
    - min_max_df: 包含 'MinValue' 和 'MaxValue' 的 DataFrame。
    """
    # 读取CSV文件
    data = pd.read_csv(csv_path)

    # 提取 'MinValue' 和 'MaxValue'
    min_max_df = pd.DataFrame({
        'MinValue': data['MinValue'].values,
        'MaxValue': data['MaxValue'].values
    })

    return min_max_df
def normalize_data(data, min_max_df):
    """
    对数据进行归一化处理，模拟 sklearn 的 MinMaxScaler 行为。
    
    参数：
    - data: 待归一化的特征数据 (NumPy ndarray 或 Pandas DataFrame，无列名)。
    - min_max_df: 包含每列的最小值和最大值的 DataFrame，其中行号应与数据的列索引对应。
    
    返回：
    - 归一化后的数据 (ndarray)。
    """
    # 确保输入数据是 NumPy 数组
    if isinstance(data, pd.DataFrame):
        data = data.values

    # 检查数据和 min_max_df 的列数是否一致
    if data.shape[1] != len(min_max_df):
        raise ValueError("数据列数与 min_max_df 行数不一致！")

    # 提取最小值和最大值
    min_values = min_max_df['MinValue'].values
    max_values = min_max_df['MaxValue'].values

    # 计算分母，并检查分母是否为零
    denominator = max_values - min_values
    zero_mask = denominator == 0  # 标记分母为零的列

    # 避免分母为零的情况
    denominator[zero_mask] = 1  # 对分母为零的列设置为 1，避免除零

    # 归一化公式： (x - min) / (max - min)
    normalized_data = (data - min_values) / denominator

    # 模拟 sklearn 行为：对于分母为零的列，归一化值设置为 0
    normalized_data[:, zero_mask] = 0

    return normalized_data


# 模型定义
class ProposedModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ProposedModel, self).__init__()
        
        # 第一层并行卷积
        self.conv1_1 = nn.Conv1d(1, 64, kernel_size=1)
        self.conv1_3 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        
        # 第二层并行卷积
        self.conv2_1 = nn.Conv1d(192, 64, kernel_size=1)
        self.conv2_3 = nn.Conv1d(192, 64, kernel_size=3, padding=1)
        self.conv2_5 = nn.Conv1d(192, 64, kernel_size=5, padding=2)
        
        # 最后一层卷积
        self.conv3 = nn.Conv1d(192, 64, kernel_size=7, padding=3)
        
        # 计算卷积后展平维度
        self.flatten_dim = 64 * input_dim

        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # 第一层并行卷积
        x1_1 = self.relu(self.conv1_1(x))
        x1_3 = self.relu(self.conv1_3(x))
        x1_5 = self.relu(self.conv1_5(x))
        x1 = torch.cat([x1_1, x1_3, x1_5], dim=1)
        
        # 第二层并行卷积
        x2_1 = self.relu(self.conv2_1(x1))
        x2_3 = self.relu(self.conv2_3(x1))
        x2_5 = self.relu(self.conv2_5(x1))
        x2 = torch.cat([x2_1, x2_3, x2_5], dim=1)
        
        # 最后一层卷积
        x3 = self.relu(self.conv3(x2))
        
        # 展平并通过全连接层
        x_flat = self.flatten(x3)
        x_fc1 = self.relu(self.fc1(x_flat))
        x_fc2 = self.relu(self.fc2(x_fc1))
        x_fc3 = self.fc3(x_fc2)
        
        return x_fc2, x_fc3

# 定义类别与权重的映射关系
CATEGORY_WEIGHTS = {
    0: 5,   # Benign
    4: 20,  # Reconnaissance
    2: 22,  # Fuzzers
    3: 24,  # Generic
    1: 26,  # Exploits
    5: 28   # Shellcode
}

def embedding_generate_with_weights_and_min_max(model, data, min_max_df):
    """
    对输入数据进行归一化，并通过模型生成类别预测、嵌入向量和类别权重。
    
    参数：
    - model: 训练好的 PyTorch 模型。
    - data: 待处理的数据 (NumPy ndarray 或 Pandas DataFrame，无列名)。
    - min_max_df: 包含每列最小值和最大值的 DataFrame，用于归一化。
    
    返回：
    - original_data: 原始数据 (ndarray)。
    - embeddings: 数据的嵌入向量 (ndarray)。
    - weights: 每条数据的类别权重 (ndarray)。
    """
    # 确保模型处于评估模式
    model.eval()

    # 对数据进行归一化
    normalized_data = normalize_data(data, min_max_df)
    
    # 将数据转换为 PyTorch 张量
    input_tensor = torch.from_numpy(normalized_data).float()
    
    # 禁用梯度计算
    with torch.no_grad():
        # 通过模型进行前向传播
        embeddings, logits = model(input_tensor)
        
        # 计算类别预测
        predictions = torch.argmax(logits, dim=1).numpy()

        embeddings = embeddings.numpy()

        # 根据预测类别计算权重
        weights = np.array([CATEGORY_WEIGHTS.get(pred, 0) for pred in predictions])  # 默认权重为 0
    
    # 返回原始数据、嵌入向量和权重
    return embeddings, weights

def embedding_generate_with_weights(model, data):
    """
    通过模型生成类别预测、嵌入向量和类别权重。
    
    参数：
    - model: 训练好的 PyTorch 模型。
    - data: 已归一化的数据 (NumPy ndarray 或 Pandas DataFrame，无列名)。
    
    返回：
    - embeddings: 数据的嵌入向量 (ndarray)。
    - weights: 每条数据的类别权重 (ndarray)。
    """
    # 确保模型处于评估模式
    model.eval()

    # 将数据转换为 PyTorch 张量
    input_tensor = torch.from_numpy(data).float()
    
    # 禁用梯度计算
    with torch.no_grad():
        # 通过模型进行前向传播
        embeddings, logits = model(input_tensor)
        
        # 计算类别预测
        predictions = torch.argmax(logits, dim=1).numpy()

        embeddings = embeddings.numpy()

        # 根据预测类别计算权重
        weights = np.array([CATEGORY_WEIGHTS.get(pred, 0) for pred in predictions])  # 默认权重为 0
    
    # 返回嵌入向量和权重
    return embeddings, weights

class UserEmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_classes):
        super(UserEmbeddingNet, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, embedding_dim),
            nn.PReLU()
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.classifier(embedded)
        return embedded, output


def get_random_data_label(csv_path):
    """
    从CSV文件中随机挑选一行，提取其数据部分和标签。

    参数：
    - csv_path: CSV文件路径。

    返回：
    - data: 符合指定格式的 NumPy 数组 (2D)。
    - label: 对应的标签 (int 或其他类型)。
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 随机选择一行
    random_row = df.sample(n=1)
    # 提取数据部分（第二列到倒数第二列）

    data = random_row.iloc[:, 1:-1].values.astype(float)

    # 提取标签（最后一列）
    label = random_row.iloc[:, -1].values[0]

    # 将数据部分包装为2D NumPy数组
    data = np.expand_dims(data.flatten(), axis=0)

    return random_row, data, label
# 示例使用
if __name__ == "__main__":
    data_path = 'dataset\sampled_data.csv'
    min_max_path = 'dataset\min_max_values.csv'
    model_path = 'model\model_all_data_2024-11-20-17-59.pth'
    original_data, data, label = get_random_data_label(data_path)

    # 提取最小值和最大值
    min_max_df = extract_min_max(min_max_path)

    # 模型初始化 (假设输入维度为 data.shape[1]，类别数为6)
    input_dim = data.shape[1]
    num_classes = 6
    embedding_dim = 128
    # model = ProposedModel(input_dim=input_dim, num_classes=num_classes)
    model=UserEmbeddingNet(input_dim, embedding_dim,num_classes)
    model.load_state_dict(torch.load(model_path))

    # 调用函数生成嵌入和权重
    embeddings, weights = embedding_generate_with_weights(model, data, min_max_df)

    # 打印结果
    print("原始数据：", original_data)
    print("嵌入向量：", embeddings)
    print("类别权重：", weights)