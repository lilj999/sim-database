

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time
import time

from tqdm import tqdm
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_and_preprocess_data(label_path, data_path, num_samples=446831):
    data = pd.read_csv(data_path)
    label = pd.read_csv(label_path)

    X = data.iloc[0:num_samples, :].values  # 特征数据
    y = torch.from_numpy(label['Label'].to_numpy()[0:num_samples])

    # 使用 Min-Max 归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # return X, y
    return X,X_normalized, y

def create_dataloader(X,y, batch_size=256):
    # X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    # y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    # X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    # y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    X_tensor= torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
def split_data(X_normalized, y, test_size=0.05, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # 保证每类比例一致
    )
    return X_train, X_test, y_train, y_test
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



def visualize_embeddings2(model, original_data_loader, proceed_data_loader):
    embeddings = []
    original = []
    labels = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for X, y in proceed_data_loader:
            X, y = X.to(device), y.to(device)
            embedding, _ = model(X)
            embeddings.append(embedding.cpu())
            labels.append(y.cpu())
        for X, y in original_data_loader:
            original.append(X.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    original = torch.cat(original, dim=0)
    labels = torch.cat(labels, dim=0)

    print(embeddings.shape, original.shape, labels.shape)
    # 过滤掉 label=7 的数据
    mask = (labels != 7)
    all_embeddings = embeddings[mask]
    all_labels = labels[mask]
    all_X = original[mask]

    # 使用 PCA 降维到 64 维
    pca = PCA(n_components=64)
    embeddings_pca_e = pca.fit_transform(all_embeddings.numpy())
    embeddings_pca_x = pca.fit_transform(all_X.numpy())
    print(embeddings_pca_e.shape)

    # 使用 t-SNE 将降维结果再降到 2 维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=1000, n_iter=300)
    embeddings_2d_e = tsne.fit_transform(embeddings_pca_e)
    embeddings_2d_x = tsne.fit_transform(embeddings_pca_x)

    unique_labels = np.unique(all_labels.numpy())
    cat_dic = {'0': 'Benign', '1': 'Exploits', '2': 'Fuzzers', '3': 'Generic',
               '4': 'Reconnaissance', '5': 'Shellcode'}

    # 绘制第一个图：原始数据 t-SNE
    plt.figure(figsize=(8, 8))
    for i in unique_labels:
        idx = all_labels == i
        cat = cat_dic.get(str(int(i)), f"Class {i}")
        plt.scatter(
            embeddings_2d_x[idx, 1], embeddings_2d_x[idx, 0],
            label=f'{cat}', s=20, alpha=0.7
        )
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('Original Traffic Data (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

    # 绘制第二个图：嵌入向量 t-SNE
    plt.figure(figsize=(8, 8))
    for i in unique_labels:
        idx = all_labels == i
        cat = cat_dic.get(str(int(i)), f"Class {i}")
        plt.scatter(
            embeddings_2d_e[idx, 1], embeddings_2d_e[idx, 0],
            label=f'{cat}', s=20, alpha=0.7
        )
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('Generated Embedding Vectors (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()



def visualize_embeddings(model, original_data_loader,proceed_data_loader):
    embeddings=[]
    original=[]
    labels=[]
    model.eval()
    model.to(device)
    with torch.no_grad():
        for X, y in proceed_data_loader:
            X, y = X.to(device), y.to(device)
            embedding,_ = model(X)
            embeddings.append(embedding.cpu())
            labels.append(y.cpu())
        for X, y in original_data_loader:
            original.append(X.cpu())

    embeddings=torch.cat(embeddings,dim=0)
    original=torch.cat(original,dim=0)
    labels=torch.cat(labels,dim=0)
    # 拼接所有批次的数据

    print(embeddings.shape, original.shape,labels.shape)
    # 过滤掉 label=0 的数据

    mask = (labels != 7)  # 保留 label 不等于 0 的数据
    # mask=all_labels
    all_embeddings = embeddings[mask]
    all_labels = labels[mask]
    all_X = original[mask]

    # 使用 PCA 降维到 50 维
    pca = PCA(n_components=64)
    embeddings_pca_e = pca.fit_transform(all_embeddings.numpy())
    embeddings_pca_x = pca.fit_transform(all_X.numpy())
    print(embeddings_pca_e.shape)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=1000, n_iter=300)
    embeddings_2d_e = tsne.fit_transform(embeddings_pca_e)
    embeddings_2d_x = tsne.fit_transform(embeddings_pca_x)
    # embeddings_2d = tsne.fit_transform(all_embeddings.numpy())

    # plt.figure(figsize=(10, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    unique_labels = np.unique(all_labels.numpy())
    cat_dic={'0':'Benign', '1':'Exploits', '2':'Fuzzers','3':' Generic','4':'Reconnaissance','5':'Shellcode'}

    for i in unique_labels:
        idx = all_labels == i
        print(i,type(i))
        cat = cat_dic.get(str(int(i)))
        # plt.subplot(1, 2, 2,figsize=(8,8))
        ax1.scatter(
            # embeddings_pca[idx, 0], embeddings_pca[idx, 2],
            embeddings_2d_x[idx, 1], embeddings_2d_x[idx, 0],
            label=f'{cat}', s=20, alpha=0.7
        )

        ax2.scatter(
            # embeddings_pca[idx, 0], embeddings_pca[idx, 2],
            embeddings_2d_e[idx, 1], embeddings_2d_e[idx, 0],
            label=f'{cat}', s=20, alpha=0.7
        )
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax1.legend()
    ax2.legend()
    ax1.set_title('Original Traffic Data (t-SNE)')
    ax2.set_title('Generated Embedding Vectors (t-SNE)')
    ax1.set_xlabel('t-SNE Component 1')
    ax2.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax2.set_ylabel('t-SNE Component 2')
    # fig.tight_layout()
    # fig.savefig()

    plt.show()


# 注意：请确保在实际环境中正确设置device参数，例如 'cuda' 或 'cpu'


def main():
    label_path = './dataset/filtered_labels.csv'
    data_path = './dataset/filtered_data.csv'
    import time
    local_time = time.localtime()
    time = time.strftime("%Y-%m-%d-%H-%M", local_time)
    print(time)

    X,X_normalized, y = load_and_preprocess_data(label_path, data_path)

    X_train, X_pro, y_train, y_test = split_data(X_normalized, y)
    X_train, X_ori, y_train, y_test = split_data(X, y)

    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y))
    print(num_classes)
    embedding_dim = 128
    model = UserEmbeddingNet(input_dim, embedding_dim, num_classes)
    # model.load_state_dict(torch.load(f"./model/model_all_data_2024-11-20-17-59.pth", map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(f"./model/model_all_data_2024-11-20-17-59.pth"))

    ori_data_loader=create_dataloader(X_ori, y_test, batch_size=256)
    pro_data_loader=create_dataloader(X_pro, y_test, batch_size=256)

    visualize_embeddings2(model,ori_data_loader,pro_data_loader)


if __name__ == '__main__':
    main()