import random
from collections import defaultdict

class SimHashDatabase:
    def __init__(self, num_bits=64, num_segments=4, hamming_threshold=3):
        """
        初始化 SimHash 数据库。
        :param num_bits: SimHash 的位数（嵌入向量的降维后长度，默认 64 位）。
        :param num_segments: 分段数量（默认 4 段）。
        :param hamming_threshold: 汉明距离阈值（默认 3）。
        """
        self.num_bits = num_bits
        self.num_segments = num_segments
        self.hamming_threshold = hamming_threshold
        self.segment_length = num_bits // num_segments
        self.hash_tables = [defaultdict(list) for _ in range(num_segments)]
        self.data_store = {}  # 存储完整的嵌入向量

    def _hash_feature(self, feature):
        """
        对关键词特征进行固定的哈希编码。
        :param feature: 关键词特征。
        :return: 哈希值（64 位整数）。
        """
        return hash(feature) & ((1 << self.num_bits) - 1)

    def _generate_simhash(self, features):
        """
        基于权重对特征进行加权，叠加生成最终嵌入向量，并降维生成 SimHash。
        :param features: {关键词: 权重} 的特征字典。
        :return: SimHash 签名（二进制字符串）。
        """
        hash_vector = [0] * self.num_bits
        for feature, weight in features.items():
            feature_hash = self._hash_feature(feature)
            for i in range(self.num_bits):
                if feature_hash & (1 << i):
                    hash_vector[i] += weight
                else:
                    hash_vector[i] -= weight
        return ''.join(['1' if x > 0 else '0' for x in hash_vector])

    def _divide_into_segments(self, simhash_signature):
        """
        将 SimHash 签名分成多个段。
        :param simhash_signature: SimHash 签名（二进制字符串）。
        :return: 分段后的列表。
        """
        return [simhash_signature[i * self.segment_length:(i + 1) * self.segment_length]
                for i in range(self.num_segments)]

    def add_to_database(self, doc_id, features):
        """
        存储阶段：根据特征生成嵌入向量和最终指纹，分段存储到桶中。
        :param doc_id: 数据的 ID。
        :param features: 数据的特征及其权重，格式为 {关键词: 权重}。
        """
        simhash_signature = self._generate_simhash(features)
        self.data_store[doc_id] = simhash_signature  # 存储完整的哈希值
        segments = self._divide_into_segments(simhash_signature)
        for i, segment in enumerate(segments):
            # 将文档 ID 和完整哈希值存入对应段的桶
            self.hash_tables[i][segment].append((doc_id, simhash_signature))

    def query(self, query_features):
        """
        查询阶段：基于特征计算 SimHash 签名，分段检索，筛选结果。
        :param query_features: 查询数据的特征及其权重。
        :return: 满足条件的文档 ID 列表和对应的汉明距离列表。
        """
        query_signature = self._generate_simhash(query_features)
        query_segments = self._divide_into_segments(query_signature)

        # 搜索阶段：逐段查询，收集候选文档
        candidate_docs = set()
        for i, segment in enumerate(query_segments):
            for doc_id, stored_signature in self.hash_tables[i].get(segment, []):
                candidate_docs.add((doc_id, stored_signature))

        # 筛选阶段：利用汉明距离对候选文档进行筛选
        doc_ids = []
        distances = []
        for doc_id, stored_signature in candidate_docs:
            distance = self._hamming_distance(query_signature, stored_signature)
            if distance <= self.hamming_threshold:
                doc_ids.append(doc_id)
                distances.append(distance)

        return doc_ids, distances

    def _hamming_distance(self, sig1, sig2):
        """
        计算两个 SimHash 签名的汉明距离。
        :param sig1: 第一个 SimHash 签名。
        :param sig2: 第二个 SimHash 签名。
        :return: 汉明距离。
        """
        return sum(c1 != c2 for c1, c2 in zip(sig1, sig2))

def main():
    # 定义特征池和固定权重范围
    FEATURE_POOL = [f"feature_{i}" for i in range(11)]
    FIXED_WEIGHTS = [1, 2, 3, 4, 5]

    # 初始化数据库
    db = SimHashDatabase(num_bits=64, num_segments=4, hamming_threshold=30)

    # 随机生成文档数据
    documents = {
        f"doc_{i}": {
            random.choice(FEATURE_POOL): random.choice(FIXED_WEIGHTS) for _ in range(5)
        }
        for i in range(100)
    }

    # 存储阶段
    for doc_id, features in documents.items():
        db.add_to_database(doc_id, features)

    # 查询阶段
    query_features = {"feature_2": 3, "feature_4": 1, "feature_8": 2}
    result = db.query(query_features)

    # 输出结果
    print("数据库中的文档及其特征：")
    for doc_id, features in documents.items():
        print(f"{doc_id}: {features}")

    print("\n查询特征:", query_features)
    print("相似文档:", result)


# 运行主函数
if __name__ == "__main__":
    main()