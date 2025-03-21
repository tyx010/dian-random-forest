import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, gini=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.gini = gini
        self.left = left
        self.right = right

class Tree:
    def __init__(self, max_depth=None, min_samples_split=2, features_n=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features_n = features_n
        self.tree = None

    def calgini(self, y):
        gini = 1
        classes = np.unique(y)
        for c in classes:
            p = np.sum(y == c) / len(y)
            gini -= p ** 2
        return gini

    def split(self, X, y, feature, threshold):
        left_samples = X[:, feature] <= threshold
        right_samples = X[:, feature] > threshold
        return X[left_samples], X[right_samples], y[left_samples], y[right_samples]

    def calmiddle(self, x):
        arr = np.sort(x)
        return (arr[:-1] + arr[1:]) / 2

    def best_split(self, X, y):
        best_feature = None
        best_threshold = None
        min_gini = np.inf

        if self.features_n is not None:
            feature_indices = np.random.choice(X.shape[1], self.features_n, replace=False)
        else:
            feature_indices = range(X.shape[1])

        for feature in feature_indices:
            thresholds = self.calmiddle(X[:, feature])
            for threshold in thresholds:
                _, _, y_left, y_right = self.split(X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gini = (len(y_left) * self.calgini(y_left) + len(y_right) * self.calgini(y_right)) / len(y)
                if gini < min_gini:
                    min_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        node = Node()
        y_classes = np.unique(y)
        
        if depth == self.max_depth or len(y) < self.min_samples_split or len(y_classes) == 1:
            node.threshold = max(y_classes, key=y.tolist().count)  
            return node
    
        node.feature, node.threshold = self.best_split(X, y)
    
        if node.feature is None or node.threshold is None:
            node.threshold = max(y_classes, key=y.tolist().count) 
            return node
    
    
        left_samples = X[:, node.feature] <= node.threshold
        right_samples = X[:, node.feature] > node.threshold
    
  
        node.left = self.build_tree(X[left_samples], y[left_samples], depth + 1)
        node.right = self.build_tree(X[right_samples], y[right_samples], depth + 1)
    
        return node

    def train(self, X, y):
        self.tree = self.build_tree(X, y)
        return self.tree

    def predict_onesample(self, x, node):
        if node.left is None and node.right is None:
            return node.threshold
        feature = node.feature
        threshold = node.threshold
        if x[feature] <= threshold:
            return self.predict_onesample(x, node.left)
        else:
            return self.predict_onesample(x, node.right)

    def predict(self, X):
        return np.array([self.predict_onesample(x, self.tree) for x in X])

class RandomForest:
    def __init__(self, trees_n=100, max_depth=None, min_samples_split=2, features_n=None):
        self.trees_n = trees_n
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features_n = features_n
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indexes = np.random.choice(n_samples, n_samples, replace=True)
        return X[indexes], y[indexes]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.trees_n):
            tree = Tree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, features_n=self.features_n)
            X_samples, y_samples = self.bootstrap_sample(X, y)
            tree.train(X_samples, y_samples)
            self.trees.append(tree)

    def predict(self, X):
        predict_results = np.array([tree.predict(X) for tree in self.trees])
        final_results = np.array([np.argmax(np.bincount(predict_results[:, i])) for i in range(X.shape[0])])
        return final_results

    def tree_feature_importance(self, tree, n_features, importance=None):
        if importance is None:
            importance = np.zeros(n_features)  # 初始化 importance 数组，维度为特征数量
        if tree.left is not None and tree.right is not None:
            importance[tree.feature] += 1 
            self.tree_feature_importance(tree.left, n_features, importance)  
            self.tree_feature_importance(tree.right, n_features, importance)  
        return importance

    def forest_feature_importance(self, X):
        n_features = X.shape[1]  
        importances = np.zeros(n_features)
        for tree in self.trees:
            importances += self.tree_feature_importance(tree.tree, n_features)  
        return importances / len(self.trees)  # 返回平均重要性

    def get_params(self, deep=True):
        return {
            'trees_n': self.trees_n,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'features_n': self.features_n
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self