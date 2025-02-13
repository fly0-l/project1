import tensorflow as tf
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv1D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold


# 预处理函数
def get_train_test_dataset(df):
    data = df.iloc[:, :df.shape[1] - 1]  # 获取特征数据
    labels = df.iloc[:, -1].astype(int)  # 获取最后标签数据并转换为整数

    # 对特征数据进行归一化处理
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)  # 直接归一化所有特征

    # 使用 ROS 进行过采样，平衡数据
    ros = RandomOverSampler(random_state=42)
    data_balanced, labels_balanced = ros.fit_resample(data_normalized, labels)

    # 调整输入形状为 Conv1D 期望的三维输入： (batch_size, timesteps, features)
    data_balanced = np.expand_dims(data_balanced, axis=-1)

    return data_balanced, np.array(labels_balanced)


# 模型训练函数
def train(X_train, y_train, n_estimators, projectname, num):
    models = []
    for _ in range(n_estimators):
        model = Sequential()
        model.add(Conv1D(128, 1, padding='same', activation='relu', input_shape=(num, 1)))  # 第一层卷积
        model.add(Dropout(0.2, seed=42))  # 防止过拟合
        model.add(Conv1D(128, 1, activation='tanh'))  # 第二层卷积
        model.add(Conv1D(128, 1, activation='tanh'))  # 第三层卷积
        model.add(Flatten())
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))  # 输出层，用于二分类

        if projectname in ['gc', 'fe']:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy',
                          metrics=['accuracy'])
        else:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 训练模型
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)  # 使用fit而非train_on_batch

        models.append(model)

        # 保存模型结构（JSON格式）
        model_json = model.to_json()
        model_filename_json = f'/root/autodl-tmp/f/加入数据采样/noNclRf/{projectname}-'+(str)(_)+'.json'
        with open(model_filename_json, 'w') as json_file:
            json_file.write(model_json)

        # 保存模型权重（H5格式）
        model_filename_h5 = f'/root/autodl-tmp/f/加入数据采样/noNclRf/{projectname}-'+(str)(_)+'.h5'
        model.save_weights(model_filename_h5)

    return models


# 模型加载函数
def load_models(projectname, n_estimators):
    from tensorflow.keras.models import model_from_json
    models = []
    for i in range(n_estimators):
        model = model_from_json(open(f'/root/autodl-tmp/f/加入数据采样/noNclRf/{projectname}-{i}.json').read())
        model.load_weights(f'/root/autodl-tmp/f/加入数据采样/noNclRf/{projectname}-{i}.h5', by_name=True)
        models.append(model)
    return models


# 模型测试函数
def test(X_test, y_test, projectname, n_estimators):
    models = load_models(projectname, n_estimators)
    ensemble_predictions = []
    for model in models:
        predictions = model.predict(X_test)
        ensemble_predictions.append(predictions)

    # 取多个模型的平均预测结果
    ensemble_prediction = np.mean(ensemble_predictions, axis=0)
    predictions = (ensemble_prediction > 0.5).astype(int)  # 转换为二分类标签

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='binary')
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    auc = roc_auc_score(y_test, ensemble_prediction)

    return accuracy, f1, precision, recall, auc


# 十折交叉验证函数
def cross_validate(data, labels, n_estimators, projectname, num_folds=10):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "f1": [], "precision": [], "recall": [], "auc": []}

    fold = 1
    for train_index, test_index in kf.split(data):
        fold += 1

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        models = train(X_train, y_train, n_estimators, projectname, X_train.shape[1])
        accuracy, f1, precision, recall, auc = test(X_test, y_test, projectname, n_estimators)

        metrics["accuracy"].append(accuracy)
        metrics["f1"].append(f1)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["auc"].append(auc)

    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}

    return avg_metrics
