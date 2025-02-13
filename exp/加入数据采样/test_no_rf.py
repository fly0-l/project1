'''
没有特征选择
有ncl
进行十折交叉验证
'''

import noRf
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
    lambda_ncl=[0.25,0.1,0.2,0.05]
    data_normalized, labels = noRf.get_train_test_dataset(i)
    # 十折交叉验证
    metrics = noRf.cross_validate(data_normalized, labels, n_estimators=n_estimators[j], lambda_ncl=lambda_ncl[j],
                                             projectname=data_names[j])

    # 打印十折交叉验证结果
    print(f'*{data_names[j]}*'*20)
    print("10-Fold Cross-Validation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    j = j+1
