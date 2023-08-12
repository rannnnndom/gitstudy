import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self, feature=None, threshold=None, label=None, left=None, middle=None, right=None):
        self.feature = feature  # 特征索引
        self.threshold = threshold  # 阈值（用于连续特征）
        self.label = label  # 叶节点的类标签
        self.left = left  # 左子树节点
        self.middle = middle  # 中间子树节点（适用于三分类）
        self.right = right  # 右子树节点

class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        if depth == self.max_depth or len(np.unique(y)) == 1:
            return TreeNode(label=np.bincount(y).argmax())

        best_feature = None
        best_threshold = None
        best_gain = -1.0

        for feature in range(num_features):
            values = X[:, feature]

            if len(np.unique(values)) == 1:
                continue

            for threshold in np.unique(values):
                left_indices = values <= threshold
                middle_indices = (values > threshold) & (values <= threshold + 1)
                right_indices = values > threshold + 1
                if np.any(left_indices) and np.any(middle_indices) and np.any(right_indices):
                    gain = self._information_gain(y, y[left_indices], y[middle_indices], y[right_indices])
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

        if best_gain == 0:
            return TreeNode(label=np.bincount(y).argmax())

        left_indices = X[:, best_feature] <= best_threshold
        middle_indices = (X[:, best_feature] > best_threshold) & (X[:, best_feature] <= best_threshold + 1)
        right_indices = X[:, best_feature] > best_threshold + 1
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        middle_tree = self._build_tree(X[middle_indices], y[middle_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_tree, middle=middle_tree, right=right_tree)

    def _information_gain(self, parent, left_child, middle_child, right_child):
        weight_parent = len(parent)
        weight_left = len(left_child)
        weight_middle = len(middle_child)
        weight_right = len(right_child)

        entropy_parent = self._entropy(parent)
        entropy_left = self._entropy(left_child)
        entropy_middle = self._entropy(middle_child)
        entropy_right = self._entropy(right_child)

        return entropy_parent - ((weight_left / weight_parent) * entropy_left + (weight_middle / weight_parent) * entropy_middle + (weight_right / weight_parent) * entropy_right)

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        class_counts = np.bincount(y)
        class_probs = class_counts / len(y)
        return -np.sum(class_probs * np.log2(class_probs + 1e-10))

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.label is not None:
            return node.label
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        elif x[node.feature] <= node.threshold + 1:
            return self._traverse_tree(x, node.middle)
        else:
            return self._traverse_tree(x, node.right)

# 示例用法
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 1.5], [1.5, 0.5]])
y = np.array([0, 1, 2, 0, 1, 2])

tree = ID3DecisionTree(max_depth=2)
tree.fit(X, y)

new_samples = np.array([[0, 1], [1, 0], [1.5, 0.5]])
predictions = tree.predict(new_samples)
print(predictions)
