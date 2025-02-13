'''
没有ncl
有特征选择
进行交叉验证
'''

import noNCL
import pandas as pd

df_fe= pd.read_csv("D:/pycharm/pythonProject2/datasets/FeatureEnvy.csv")
df_dc= pd.read_csv("D:/pycharm/pythonProject2/datasets/DataClass.csv")
df_gc= pd.read_csv("D:/pycharm/pythonProject2/datasets/GodClass.csv")
df_lm= pd.read_csv("D:/pycharm/pythonProject2/datasets/LongMethod.csv")

datasets=[df_lm,df_gc,df_dc,df_fe]
data_names=["lm","gc","dc","fe"]
j=0

for i in datasets:
    n_estimators=[3,6,5,5]
    data_normalized, labels = noNCL.get_train_test_dataset(i)
    # 十折交叉验证
    metrics = noNCL.cross_validate(data_normalized, labels, n_estimators=n_estimators[j],
                                             projectname=data_names[j])

    # 打印十折交叉验证结果
    print(f'*{data_names[j]}*'*20)
    print("10-Fold Cross-Validation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    j = j+1

