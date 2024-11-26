import random
import json
from collections import defaultdict

import random
import pydotplus
from IPython.display import Image, display
import os
os.environ["PATH"] += os.pathsep + 'E:\\Graphviz\\bin'

# 定义决策树节点类
class DecisionTreeNode:
    def __init__(self, rule=None, children=None, action=None,desc=None):
        self.rule = rule  # 规则函数
        self.desc =desc
        self.children = children if children else []  # 子节点
        self.action = action  # 如果是叶子节点，执行的动作

    def is_leaf(self):
        return self.action is not None

    def add_child(self, child):
        self.children.append(child)


# 定义规则函数
def rule_protocol(data):
    return data['protocol']


def rule_security(data):
    return 'safe' if data['security'] > 0.5 else 'unsafe'


def rule_behavior(data):
    return 'normal' if data['behavior'] < 0.5 else 'anomalous'

def rule_root(data):
    return data['rule']
# 构造决策树
def build_decision_tree():
    root = DecisionTreeNode(rule=rule_root,desc='root')

    # 协议子节点
    safe_node = DecisionTreeNode(rule=rule_security,desc='safety')
    behavior_node = DecisionTreeNode(rule=rule_behavior,desc='behavior')
    protocol_node = DecisionTreeNode(rule=rule_protocol,desc='protocol')

    # 安全性子节点
    # Benign(0)  —— 5
    # Reconnaissance(4) —— 20
    # Fuzzers(2)  —— 22
    # Generic(3)  —— 24
    # Exploits(1)  —— 26
    # Shellcode(5) —— 28
    safe_list=['Benign','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode']
    protocol_list=['TCP','UDP']
    behavior_list=['behavior1','behavior2','behavior3']
    for item in safe_list:
        safe_node.add_child(DecisionTreeNode(action=f'Process {item}',desc=f'Process {item}'))
    for item in protocol_list:
        protocol_node.add_child(DecisionTreeNode(action=f'Process {item}',desc=f'Process {item}'))
    for item in behavior_list:
        behavior_node.add_child(DecisionTreeNode(action=f'Process {item}',desc=f'Process {item}'))



    # 添加子节点到根节点
    root.add_child(safe_node)  # HTTP
    root.add_child(protocol_node)
    root.add_child(behavior_node)  # HTTPS

    return root


# 决策过程
def make_decision(tree, data, path=None):
    if path is None:
        path = []
    if tree.is_leaf():
        path.append(f"Action: {tree.action}")
        return path
    rule_value = tree.rule(data)
    path.append(f"Rule: {tree.rule.__name__} -> {rule_value}")

    # 根据规则值选择对应子节点
    for child in tree.children:
        if rule_value == 'http' and child.rule == rule_security:
            return make_decision(child, data, path)
        elif rule_value == 'https' and child.rule == rule_behavior:
            return make_decision(child, data, path)
    return path








def visualize_tree(tree, graph=None, parent_name=None, edge_label="", show_id=False):
    """
    Visualize a decision tree structure using pydotplus.

    Parameters:
        tree (object): The root node of the decision tree. The node should have attributes:
                       - is_leaf(): Method to check if the node is a leaf.
                       - desc (str): A description of the node.
                       - children (list): List of child nodes (empty if leaf).
                       - condition (optional): Condition leading to the child node.
        graph (pydotplus.Dot): The graph object (used for recursion). Default is None.
        parent_name (str): The parent node's name. Default is None.
        edge_label (str): The label for the edge connecting the parent and the current node.
        show_id (bool): Whether to display node IDs for debugging purposes. Default is False.

    Returns:
        pydotplus.Dot: A pydotplus graph object representing the tree.
    """
    if graph is None:
        graph = pydotplus.Dot(
            graph_type='digraph',  # Use directed graph for clarity in hierarchy
            bgcolor='white',  # Set background color to white for academic presentation
            rankdir='TB',  # Top to bottom layout
            splines='true',  # Smooth edges
            nodesep=0.7,  # Horizontal spacing between nodes
            ranksep=1.0,  # Vertical spacing between levels
            margin=1.0  # Increase margin for better padding around the graph
        )

    # Generate unique identifier for the current node
    node_id = id(tree)
    node_name = f"node_{node_id}"

    # Determine node properties (label, shape, and color)
    if tree.is_leaf():
        label = f"Leaf: {tree.desc}" + (f"\nID: {node_id}" if show_id else "")
        color = "#C3E6CB"  # Soft green for leaf nodes
        shape = 'rectangle'
    else:
        label = f"Decision: {tree.desc}" + (f"\nID: {node_id}" if show_id else "")
        color = "#FFDDC1"  # Light orange for decision nodes
        shape = 'ellipse'

    # Add the current node to the graph
    graph.add_node(
        pydotplus.Node(
            name=node_name,
            label=label,
            style='filled',
            fillcolor=color,
            shape=shape,
            fontname='Arial',
            fontsize='12',  # Smaller font size for academic style
            penwidth=1.5  # Thicker border for better visibility in print
        )
    )

    # Add an edge connecting the current node to its parent (if any)
    if parent_name:
        graph.add_edge(
            pydotplus.Edge(
                src=parent_name,
                dst=node_name,
                label=edge_label,
                fontname='Arial',
                fontsize='10',  # Smaller font size for edge labels
                penwidth=1.2,  # Thinner edge for better contrast
                color="#4B4B4B"  # Dark gray for edges
            )
        )

    # Recursively visualize child nodes
    for i, child in enumerate(tree.children):
        condition_label = (f"Condition {i + 1}: {child.condition}"
                           if hasattr(child, 'condition') else f"Condition {i + 1}")
        visualize_tree(child, graph, node_name, condition_label, show_id)

    return graph


# 生成模拟数据
def generate_data(num_samples=10):
    data = []
    for _ in range(num_samples):
        data.append({
            'rule':'safety',
            'protocol': random.choice(['http', 'https']),
            'security': random.random(),
            'behavior': random.random(),
        })
    return data


# 测试流程
tree = build_decision_tree()

# # 可视化树结构
# graph = visualize_tree(tree)
# graph.write_png("decision_tree.png")
# display(Image("decision_tree.png"))

# 生成数据并测试决策过程
test_data = generate_data(1)[0]
print("Test Data:", test_data)

decision_path = make_decision(tree, test_data)
print("Decision Path:")
for step in decision_path:
    print(step)

# 测试流程


