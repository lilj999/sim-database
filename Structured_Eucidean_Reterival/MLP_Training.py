import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time
import time

from tqdm import tqdm
from sklearn.cluster import KMeans

# 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理函数
def load_and_preprocess_data(label_path, data_path, num_samples=446831):
    data = pd.read_csv(data_path)
    label = pd.read_csv(label_path)

    X = data.iloc[0:num_samples, :].values  # 特征数据
    y = torch.from_numpy(label['Label'].to_numpy()[0:num_samples])

    # 使用 Min-Max 归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    return X_normalized, y


# 划分数据集
def split_data(X_normalized, y, test_size=0.05, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y  # 保证每类比例一致
    )
    return X_train, X_test, y_train, y_test


# 定义神经网络模型
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
# 创建 DataLoader
def create_dataloader(X_train, y_train, X_test, y_test, batch_size=256):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
def select_triplets(embeddings, labels, num_triplets=None):

    embeddings, labels,= embeddings.to(device), labels.to(device)
    batch_size = embeddings.size(0)
    triplets = []

    # 为每个样本选择一个三元组
    for i in range(batch_size):
        anchor_embedding = embeddings[i]  # 当前的anchor样本
        anchor_label = labels[i]  # 当前的标签

        # 获取正样本（同标签样本）
        positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
        positive_indices = positive_indices[positive_indices != i]  # 排除掉自己

        # 检查是否有正样本
        if positive_indices.size(0) == 0:
            # 如果没有正样本，可以选择负样本替代，或者跳过该三元组
            continue  # 跳过该三元组，或者选择负样本来替代

        positive_idx = random.choice(positive_indices.tolist())
        positive_embedding = embeddings[positive_idx]

        # 获取负样本（不同标签样本）
        negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
        if negative_indices.size(0) == 0:
            continue  # 如果没有负样本，也可以跳过该三元组，或者选择其他方法处理

        negative_idx = random.choice(negative_indices.tolist())
        negative_embedding = embeddings[negative_idx]

        # 将选择的三元组添加到列表
        triplets.append((anchor_embedding, positive_embedding, negative_embedding))

        # 如果需要限制三元组数量，则提前退出
        if num_triplets and len(triplets) >= num_triplets:
            break

    return triplets
def combined_loss(embedding, output, labels, criterion, margin=1.0):
    # 交叉熵损失
    embedding, labels,output = embedding.to(device), labels.to(device),output.to(device)
    ce_loss = criterion(output, labels)

    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)  # p=2 为欧几里得距离

    triplets=select_triplets(embedding,labels)
    try:
        anchor_embeddings, positive_embeddings, negative_embeddings = zip(*triplets)
        triplet_loss_value = triplet_loss_fn(torch.stack(anchor_embeddings),
                                             torch.stack(positive_embeddings),
                                             torch.stack(negative_embeddings))
        # print(f'ce_loss:{ce_loss},triplet_loss_value:{triplet_loss_value}')
        total_loss = 0.1 * ce_loss + 0.9* triplet_loss_value
    except:
        # print(f'ce_loss:{ce_loss},triplet_loss_value:{0}')
        total_loss = 0.1 * ce_loss

    return total_loss
# 模型训练函数
# def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=50):
#     model = model.to(device)
#     model.train()
#     for epoch in range(epochs):
#         print("training epoch {}".format(epoch))
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             embeddings, outputs = model(inputs)
#             loss = combined_loss(embeddings, outputs, labels, criterion)
#             loss.backward()
#             optimizer.step()
#
#         if (epoch + 1) % 5 == 0:
#             accuracy = evaluate_accuracy(model, train_loader)
#             print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.2f}%', end='')
#             accuracy = evaluate_accuracy(model, test_loader)
#             print(F'Test Accuracy: {accuracy:.2f}%')
#         torch.cuda.empty_cache()
#     print("Training complete. Saving model...")
#     torch.save(model.state_dict(), 'model.pth')


def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    for epoch in tqdm(range(1,epochs+1), desc="Training", unit="epoch"):
        # running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings, outputs = model(inputs)
            loss = combined_loss(embeddings, outputs, labels, criterion)
            loss.backward()
            optimizer.step()
            # running_loss += loss.item()
            # train_loader.set_postfix(loss=loss.item())

        if (epoch + 1) % 5 == 0:
            accuracy = evaluate_accuracy(model, train_loader)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.2f}%', end='')
            accuracy = evaluate_accuracy(model, test_loader)
            print(F'Test Accuracy: {accuracy:.2f}%')

        torch.cuda.empty_cache()
    import time
    local_time = time.localtime()
    time = time.strftime("%Y-%m-%d-%H-%M", local_time)
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), f'model_all_data_{time}.pth')
    print("Model saved!")
def evaluate_accuracy(model, data_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# 可视化嵌入 (PCA + t-SNE)
def visualize_embeddings(model, test_loader):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            embeddings, _ = model(inputs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    # 拼接所有批次的数据
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(all_embeddings.shape, all_labels.shape)
    # 过滤掉 label=0 的数据

    mask = (all_labels != 7) # 保留 label 不等于 0 的数据

    all_embeddings = all_embeddings[mask]
    all_labels = all_labels[mask]

    # 使用 PCA 降维到 50 维
    pca = PCA(n_components=32)
    embeddings_pca = pca.fit_transform(all_embeddings.numpy())

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=1000, n_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    # embeddings_2d = tsne.fit_transform(all_embeddings.numpy())


    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(all_labels.numpy())
    for i in unique_labels:
        idx = all_labels == i
        plt.scatter(
            # embeddings_pca[idx, 0], embeddings_pca[idx, 2],
            embeddings_2d[idx, 1], embeddings_2d[idx, 0],
            label=f'Category {i}', s=20, alpha=0.7
        )

    plt.legend()
    plt.title('User Data Embeddings (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('2d_filtered.jpg', dpi=300)
    plt.show()

# 主函数
def main():
    label_path = '../dataset/Structured_Data/filtered_labels.csv'
    data_path = '../dataset/Structured_Data/filtered_data.csv'
    import time
    local_time = time.localtime()
    time = time.strftime("%Y-%m-%d-%H-%M", local_time)
    print(time)
    X_normalized, y = load_and_preprocess_data(label_path, data_path)

    X_train, X_test, y_train, y_test = split_data(X_normalized, y)

    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y))
    print(num_classes)
    embedding_dim = 128
    model=UserEmbeddingNet(input_dim, embedding_dim,num_classes)
    model.load_state_dict(torch.load(f"../model/model_all_data_2024-11-20-17-59.pth", map_location=torch.device('cpu')))
    # model = ProposedModel(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_loader, test_loader = create_dataloader(X_train, y_train, X_test, y_test)

    # train_model(model, criterion, optimizer, train_loader, test_loader)

    test_accuracy = evaluate_accuracy(model, test_loader)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    visualize_embeddings(model, test_loader)
    #visualize_embeddings_3_with_clustering(model, test_loader)


if __name__ == '__main__':
    main()
