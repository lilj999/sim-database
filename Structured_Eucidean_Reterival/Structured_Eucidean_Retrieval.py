from utils.embedding_achieve import normalize_data, ProposedModel, CATEGORY_WEIGHTS, embedding_generate_with_weights_and_min_max, UserEmbeddingNet
from DataBase.Simple_Tree import EmbeddingTree, EmbeddingTreeNode
import torch
import pandas as pd
import numpy as np
import json
import random
from matplotlib import pyplot as plt
import os
# 定义路径
data_path = '../dataset/Structured_Data/sampled_data.csv'
min_max_path = '../dataset/Structured_Data/min_max_values.csv'
model_path = '../model/model_all_data_2024-11-20-17-59.pth'
output_dir = "./plot_set"

cat_dic = {'0': 'Benign', '1': 'Exploits', '2': 'Fuzzers', '3': 'Generic',
           '4': 'Reconnaissance', '5': 'Shellcode'}

import random


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
def process_all_rows(data_path, model, min_max_df, sample_count=1):
    """
    随机选择多个样本作为 sample，剩余数据用作节点。
    对每一行数据进行处理，生成嵌入向量和权重。

    参数：
    - data_path: CSV 文件路径，包含所有数据。
    - model: 已训练的 PyTorch 模型。
    - min_max_df: 包含每列最小值和最大值的 DataFrame。
    - sample_count: 随机选择的样本数量。

    返回：
    - samples: 随机选取的多行数据列表，每个元素是 (feature_name, embedding_vector, weight)。
    - nodes: 包含每行原始数据字符串、嵌入向量和权重的节点数据列表，每个元素是 (feature_name, embedding_vector, weight)。
    """
    nodes = []
    samples = []
    label_counter = {}

    # 读取数据文件
    df = pd.read_csv(data_path)

    # 随机选取样本索引
    sample_indices = set(random.sample(range(len(df)), min(sample_count, len(df))))

    # 遍历每一行
    for idx, row in df.iterrows():
        # 提取数据部分（假设数据在第 2 列到倒数第 2 列，标签在最后一列）
        data = row.iloc[1:-1].values.astype(float).reshape(1, -1)  # 转换为 2D 数组
        label = row.iloc[-1]  # 提取标签

        # 更新计数器
        if label not in label_counter:
            label_counter[label] = 0
        label_counter[label] += 1

        # 构造 feature_name
        feature_name = f"{label}-{label_counter[label]}"

        # 调用 embedding_generate_with_weights 处理单行数据
        embeddings, weights = embedding_generate_with_weights_and_min_max(model, data, min_max_df)

        # 转换嵌入向量和权重
        embedding_vector = embeddings.flatten().tolist()
        weight = float(weights.mean())  # 使用权重的均值作为树节点的权重

        # 如果索引在随机选中的样本集合中，作为 sample 加入 samples
        if idx in sample_indices:
            samples.append((feature_name, embedding_vector, weight))
        else:
            # 其余数据加入 nodes 列表
            nodes.append((feature_name, embedding_vector, weight))

    return samples, nodes



# 主函数
if __name__ == "__main__":
    # 提取 min-max 数据
    min_max_df = extract_min_max(min_max_path)
    # 模型初始化
    input_dim = len(min_max_df)  # 假设输入维度与 min_max 数据列数一致
    num_classes = 6  # 类别数
    # model = ProposedModel(input_dim=input_dim, num_classes=num_classes)
    embedding_dim = 128
    model = UserEmbeddingNet(input_dim, embedding_dim, num_classes)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')),)
    # model.load_state_dict(torch.load(f"model_all_data_2024-11-17-19-22.pth", map_location=torch.device('cpu')))
    model.eval()
    # 处理所有行
    samples, results = process_all_rows(data_path, model, min_max_df)
    # print(results)
    # 将结果保存为 JSON 文件
    output_path = 'processed_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    # 打印部分结果
    print("处理完成！结果示例：")
    # print(json.dumps(results[:3], indent=4))  # 打印前 3 条结果
    # 初始化 EmbeddingTree
    tree = EmbeddingTree(distance_threshold=2)
    # tree= SimHashTree(weight_threshold=10, hamming_threshold=2)
    # 将结果插入树
    print("\n[树构建] 开始将结果插入树中...")
    for feature_name, embedding, weight in results:
        # print(feature_name,embedding,weight)
        feature_name=cat_dic.get(feature_name.split('-')[0].split('.')[0])+'-sample'+feature_name.split('-')[1]
        tree.insert(feature_name, embedding, weight)

    print("\n[树构建] 树插入完成！当前树结构如下：")
    tree._print_tree(tree.root)

    # 利用 sample 数据进行检索
    print(sample[0].split("-")[0] for sample in samples)

    for sample in samples:
        print("\n[检索] 利用 sample 数据进行检索...")
        print(f'正在检索类别为{sample[0].split("-")[0]}的sample')
        query_embedding = sample[1]  # 使用 sample 的嵌入向量作为查询
        search_results = tree.search(query_embedding)

        print("\n[检索] 检索结果：")
        for feature_name, weight, distance in search_results[0:5]:
            print(f"  - 节点: {feature_name}, Weight: {weight}, 距离: {distance}")
    # 绘制检索结果的距离分布图
        feature_names = [result[0] for result in search_results]
        distances = [result[2] for result in search_results]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建目录

    # 图像文件路径
    output_file = os.path.join(output_dir, "sample_search_distances.png")

    # print(feature_names)
    # print(distances)
    # # 绘制图像
    # plt.figure(figsize=(10, 6))
    # plt.bar(feature_names, distances, color='skyblue')
    # plt.xlabel("Queried data")
    # plt.ylabel("Distance to Query")
    # plt.title(f"Distance of Each Feature to {feature_names[0]}")
    # plt.xticks(rotation=45)
    # plt.tight_layout()

    categories = [feature_name.split('-')[0] for feature_name in feature_names]
    # print(categories)
    # for i in range(len(categories)):
    #
    #     temp=cat_dic.get(categories[i][-1])
    #     categories[i]=temp
    #     # print(temp)
    print(categories)

    unique_categories = sorted(list(set(categories)))
    # print(unique_categories)

    # colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
    category_colors = {
        0: '#BFDFD2',
        1: '#EFCE87',
        2: '#257D8B',
        3: '#EAA558',
        4: '#68BED9',
        5: '#ED8D5A'
    }
    category_color_map = {cat: category_colors[i] for i, cat in enumerate(unique_categories)}
    bar_colors = [category_color_map[cat] for cat in categories]

    # 绘图

    plt.figure(figsize=(12, 8))
    if len(feature_names) >= 30:
        feature_names=feature_names[0:30]
        distances=distances[0:30]

    bars = plt.bar(feature_names, distances, color=bar_colors)

    # 添加数据标签
    for bar, dist in zip(bars, distances):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{dist:.2f}', ha='center', va='bottom', fontsize=10)

    # 添加图例
    handles = [plt.Line2D([0], [0], color=category_color_map[cat], lw=4)
               for cat in unique_categories]
    plt.legend(handles, unique_categories, title="Categories", fontsize=10, title_fontsize=12)

    # 设置标题和轴标签
    cat_dic = {'0': 'Benign', '1': 'Exploits', '2': 'Fuzzers', '3': 'Generic',
               '4': 'Reconnaissance', '5': 'Shellcode'}
    cat = cat_dic.get(str(int(float(samples[0][0].split('-')[0]))))
    plt.title(f"Distance of Each Feature to the Sample of {cat}", fontsize=14, weight='bold')
    plt.xlabel("Queried data", fontsize=12)
    plt.ylabel("Distance to Query", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # 调整布局
    plt.tight_layout()




    # 保存图像到文件
    plt.savefig(output_file)
    print(f"图像已保存到: {output_file}")
    plt.show()
    tf=input('write ture or fales'+'\n')
    if tf:
        with open('samples.txt', 'a+', encoding='utf-8') as file:
            # 写入一行文本到文件中
            file.write('\n'+str(samples[0]))
