from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import Random_forest
from sklearn.metrics import accuracy_score, classification_report


iris = fetch_ucirepo(id=53)


X = iris.data.features
y = iris.data.targets
# 将标签从字符串转换为整数
y = y['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# 确保 y 是整数类型
if y.dtype == 'object':
    unique_classes = np.unique(y)
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
    y = y.map(class_mapping).astype(int)

# 确保所有特征列的数据类型为浮点数
X = X.astype(float)

print("Missing values in features:\n", X.isnull().sum())
print("Missing values in labels:\n", y.isnull().sum())

# 处理缺失值（如果有）
if X.isnull().any().any() or np.isnan(y).any():
    print("Handling missing values...")
    X = X.fillna(X.mean())  # 用均值填充缺失值
    y = y.fillna(y.mode().iloc[0])  # 用众数填充缺失值


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


n_features = X_train.shape[1]
features_n = int(np.sqrt(n_features))  # 使用 sqrt(n_features) 作为特征子集大小
rf = Random_forest.RandomForest(trees_n=10000, max_depth=5, min_samples_split=2, features_n=features_n)

# 使用 K 折交叉验证评估模型
kfold = KFold(n_splits=5, shuffle=True, random_state=42)  
cv_scores = cross_val_score(rf, X_train.values, y_train.values.ravel(), cv=kfold, scoring='accuracy')

# 输出交叉验证结果
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", np.mean(cv_scores))

rf.fit(X_train.values, y_train.values.ravel())#训练


y_pred = rf.predict(X_test.values)#测试

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")


print("Classification Report:")
print(classification_report(y_test, y_pred))

importance = rf.forest_feature_importance(X_train.values)
plt.bar(range(len(importance)), importance)
plt.xticks(range(len(importance)), X.columns, rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()

joblib.dump(rf, 'random_forest_model.pkl')
print("Model saved to random_forest_model.pkl")