import random
import numpy as np
from matplotlib import pyplot as plt

class EmbeddingTreeNode:
    def __init__(self, feature_name, weight, embedding):
        self.feature_name = feature_name
        self.embedding = embedding
        self.weight = weight
        self.left = None
        self.right = None
        self.parent = None


class EmbeddingTree:
    def __init__(self, distance_threshold):
        self.root = None
        self.distance_threshold = distance_threshold  # 距离阈值

    def embedding_distance(self, embedding1, embedding2):
        """计算两个嵌入向量之间的欧氏距离"""
        return np.linalg.norm(np.array(embedding1) - np.array(embedding2))


    def _print_tree(self, node, depth=0, side="Root"):
        """递归打印二叉树的结构"""
        if node is not None:
            #print(" " * (depth * 4) + f"{side} - {node.feature_name} (Weight: {node.weight})")
            #self._print_tree(node.left, depth + 1, "Left")
            #self._print_tree(node.right, depth + 1, "Right")
            pass

    def insert(self, feature_name, embedding, weight):
        # print(f"\n[插入] 开始插入节点: {feature_name}, Weight: {weight}")
        # print("[插入] 插入前的树结构:")
        self._print_tree(self.root)

        new_node = EmbeddingTreeNode(feature_name, weight, embedding)

        if self.root is None:
            self.root = new_node
            #print(f"[插入] 根节点为空，将 {feature_name} 作为根节点插入.")
            return

        # 向树中插入新节点
        self._insert_node(new_node)

        #print("[插入] 插入后的树结构:")
        #self._print_tree(self.root)

    def _insert_node(self, new_node):
        # 将新节点插入到树的底部
        self._insert_to_bottom(new_node)
        self._heapify_up(new_node)

    def _insert_to_bottom(self, new_node):
        # 将新节点插入到树的最底部（即最后一层）
        current = self.root
        queue = [current]
        while queue:
            node = queue.pop(0)
            if node.left is None:
                node.left = new_node
                new_node.parent = node  # 设置父节点
                #print(f"[插入] 插入到 {node.feature_name} 的左子树: {new_node.feature_name}")
                return
            else:
                queue.append(node.left)

            if node.right is None:
                node.right = new_node
                new_node.parent = node  # 设置父节点
                #print(f"[插入] 插入到 {node.feature_name} 的右子树: {new_node.feature_name}")
                return
            else:
                queue.append(node.right)

    def _heapify_up(self, new_node):
        # 上浮操作，确保堆的性质得到维护
        current = new_node
        while current != self.root and current.weight > current.parent.weight:
            # 交换当前节点与父节点
            #print(f"[上浮] 当前节点: {current.feature_name}, 父节点: {current.parent.feature_name}")
            self._swap(current, current.parent)
            current = current.parent

    def _swap(self, node1, node2):
        # 交换节点1和节点2的值
        node1.feature_name, node2.feature_name = node2.feature_name, node1.feature_name
        node1.weight, node2.weight = node2.weight, node1.weight
        node1.embedding, node2.embedding = node2.embedding, node1.embedding

    def search(self, query_embedding):
        #print(f"\n[查询] 开始查询，查询特征嵌入: {query_embedding}")

        results = []

        def _search_recursive(node):
            if node is None:
                return
            distance = self.embedding_distance(query_embedding, node.embedding)
            # print(
            #     f"[查询] 当前节点: {node.feature_name}, Weight: {node.weight}, 距离: {distance}")
            if distance <= self.distance_threshold:
                # print(f"[查询] 节点 {node.feature_name} 符合查询条件，添加到结果集中.")
                results.append((node.feature_name, node.weight, distance))
            _search_recursive(node.left)
            _search_recursive(node.right)

        _search_recursive(self.root)

        # 返回符合条件的结果
        sorted_results = sorted(results, key=lambda x: (x[2], -x[1]))  # 按距离和权重排序
        length = len(sorted_results)
        #print(f"[查询] 查询完成，共 {length} 项符合条件，结果排序如下:")
        for result in sorted_results:
            pass
            #print(f"  - 节点: {result[0]}, Weight: {result[1]}, 距离: {result[2]}")
        # self._print_tree(self.root)
        return sorted_results


def main():
    # 初始化树
    simhash_tree = EmbeddingTree(distance_threshold=100)

    # 生成并插入31条随机特征数据
    feature_vectors = []
    for i in range(50):
        # 直接生成嵌入向量
        feature_vector = [random.uniform(1.0, 100.0) for _ in range(5)]  # 随机生成浮点数嵌入向量
        feature_vectors.append(feature_vector)

        # 随机生成权重
        weight = random.randint(1, 30)
        # 插入到树中
        simhash_tree.insert(f"user_behavior_{i}", feature_vector, weight)

    # 查询特征嵌入向量
    query_vector = [random.uniform(1.0, 100.0) for _ in range(5)]
    print(query_vector)
    query_vector = [1,5,8,6,7]
    # 示例查询向量
    results = simhash_tree.search(query_vector)
    # print(results)
    if len(results) < 2:
        print("查询结果不足两项，无法打印详细信息")
        return

    # 输出查询结果
    # print(results[0])
    # print(results[0][0].split("_")[-1])
    # print(feature_vectors[int(results[0][0].split("_")[-1])])
    # print(results[1])
    # print(results[1][0].split("_")[-1])
    # print(feature_vectors[int(results[1][0].split("_")[-1])])

    # 绘制查询结果的距离分布图
    feature_names = [result[0] for result in results]
    distances = [result[2] for result in results]
    weights = [result[1] for result in results]

    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, distances, color='skyblue')
    plt.xlabel("Feature Name")
    plt.ylabel("Distance to Query")
    plt.title("Distance of Each Feature to Query Feature Vector")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # 保存图像到文件
    plt.savefig("./plot_set/EuclideanDistance.png")
    plt.show()


if __name__ == "__main__":
    main()