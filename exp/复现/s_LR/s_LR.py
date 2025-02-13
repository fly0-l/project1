from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score
import pandas as pd
import numpy as np

# 加载数据集
data = pd.read_csv("/root/autodl-tmp/f/datasets/stacking/FeatureEnvy_Clean_GainRatio.csv")  # 假设存在一个 CSV 数据集
X = data.iloc[:, :-1]  # 提取特征数据（所有列除了最后一列）
y = data.iloc[:, -1]   # 提取标签数据（最后一列）

# 将数据集分为两部分：一部分用于基分类器训练，另一部分用于元分类器训练
X_train_base, X_meta, y_train_base, y_meta = train_test_split(X, y, test_size=0.5, random_state=42)

# 定义基分类器（14种单独分类器）
base_models = [
    ('dt', DecisionTreeClassifier(random_state=42)),                   # 决策树
    ('svm_lin', SVC(kernel='linear', probability=True, random_state=42)), # 线性支持向量机
    ('svm_poly', SVC(kernel='poly', probability=True, random_state=42)), # 多项式支持向量机
    ('svm_sig', SVC(kernel='sigmoid', probability=True, random_state=42)), # Sigmoid 支持向量机
    ('svm_rbf', SVC(kernel='rbf', probability=True, random_state=42)),   # RBF 支持向量机
    ('nb_b', BernoulliNB()),                                             # 伯努利朴素贝叶斯
    ('nb_g', GaussianNB()),                                              # 高斯朴素贝叶斯
    ('nb_m', MultinomialNB()),                                           # 多项式朴素贝叶斯
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),           # 逻辑回归
    ('mlp', MLPClassifier(random_state=42)),                              # 多层感知机
    ('sgd', SGDClassifier(random_state=42)),                             # 随机梯度下降
    ('gp', GaussianProcessClassifier(random_state=42)),                  # 高斯过程分类器
    ('knn', KNeighborsClassifier()),                                      # K 最近邻
    ('lda', LinearDiscriminantAnalysis())                                 # 线性判别分析
]

# 定义元分类器（逻辑回归）
meta_model = LogisticRegression()

# 构建 Stacking 模型
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=10)

# 在基分类器上进行训练
stacking_model.fit(X_train_base, y_train_base)

# 获取基分类器的输出并创建新的训练集用于元分类器
base_predictions = stacking_model.transform(X_meta)

# 使用元分类器进行训练
meta_model.fit(base_predictions, y_meta)

# 使用分层 10 折交叉验证评估模型性能
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 定义 F1 和 AUC 分数
f1_scorer = make_scorer(f1_score, average='binary')
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

# 计算准确率、F1 分数和 AUC 分数
acc_scores = cross_val_score(stacking_model, X, y, cv=stratified_kfold, scoring='accuracy')
f1_scores = cross_val_score(stacking_model, X, y, cv=stratified_kfold, scoring=f1_scorer)
auc_scores = cross_val_score(stacking_model, X, y, cv=stratified_kfold, scoring=auc_scorer)

# 输出交叉验证的准确率、F1 分数和 AUC 分数
print(f"Cross-validation accuracy scores: {acc_scores}")
print(f"Mean accuracy: {acc_scores.mean():.4f}")
print(f"Standard deviation: {acc_scores.std():.4f}")

print(f"Cross-validation F1 scores: {f1_scores}")
print(f"Mean F1 score: {f1_scores.mean():.4f}")
print(f"Standard deviation: {f1_scores.std():.4f}")

print(f"Cross-validation AUC scores: {auc_scores}")
print(f"Mean AUC score: {auc_scores.mean():.4f}")
print(f"Standard deviation: {auc_scores.std():.4f}")

# 将平均结果保存到 DataFrame
average_results = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score', 'AUC Score'],
    'Mean Score': [acc_scores.mean(), f1_scores.mean(), auc_scores.mean()],
    'Standard Deviation': [acc_scores.std(), f1_scores.std(), auc_scores.std()]
})

# 将结果四舍五入到四位小数
average_results = average_results.round(4)

# 将平均结果保存到 Excel 文件
output_path = '/root/autodl-tmp/fe_1_results.xlsx'  # 设置输出文件路径
average_results.to_excel(output_path, index=False)

print(f"Average results saved to {output_path}")
