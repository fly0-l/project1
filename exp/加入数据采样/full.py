import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv1D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
# 预处理函数
from imblearn.over_sampling import RandomOverSampler  # 引入ROS库

# 预处理函数
def get_train_test_dataset(df):
    data = df.iloc[:, :df.shape[1] - 1]  # 获取特征数据
    labels = df.iloc[:, -1].astype(int)  # 获取最后标签数据并转换为整数

    # 使用基于树模型的特征选择
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(data, labels)
    selector = SelectFromModel(clf, threshold="mean", prefit=True)
    data_selected = selector.transform(data)

    # 对特征数据进行归一化处理
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_selected)  # 归一化特征数据

    # 使用 ROS 进行过采样，平衡数据
    ros = RandomOverSampler(random_state=42)
    data_balanced, labels_balanced = ros.fit_resample(data_normalized, labels)

    # 调整输入形状为 Conv1D 期望的三维输入： (batch_size, timesteps, features)
    data_balanced = np.expand_dims(data_balanced, axis=-1)

    return data_balanced, labels_balanced



# 模型训练函数
def train(X_train, y_train, n_estimators, lambda_ncl, projectname, num):
    models = []
    for _ in range(n_estimators):
        model = Sequential()
        model.add(Conv1D(128, 1, padding='same', activation='relu', input_shape=(num, 1)))  # 第一层卷积
        model.add(Dropout(0.2,seed=42))  # 防止过拟合
        model.add(Conv1D(128, 1, activation='tanh'))  # 第二层卷积
        model.add(Conv1D(128, 1, activation='tanh'))  # 第三层卷积
        model.add(Flatten())
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))  # 输出层，用于二分类

        if projectname in ['gc', 'fe']:
           model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # 训练模型
        for epoch in range(10):  # 训练10个周期
            for i in range(0, len(X_train), 32):  # 以批大小为32进行训练
                X_batch = X_train[i:i + 32]
                y_batch = y_train[i:i + 32]

                if len(X_batch) < 32:
                    continue  # 跳过不完整的批次

                predictions = model.predict(X_batch)
                # 计算负相关损失
                loss = negative_correlation_loss(y_batch, predictions, [], lambda_ncl)
                # 更新模型
                model.train_on_batch(X_batch, y_batch)

        models.append(model)

        # 保存模型结构（JSON格式）
        model_json = model.to_json()
        model_filename_json = f'D:/pycharm/pythonProject2/加入数据采样/f/{projectname}-'+(str)(_)+'.json'
        with open(model_filename_json, 'w') as json_file:
            json_file.write(model_json)

        # 保存模型权重（H5格式）
        model_filename_h5 = f'D:/pycharm/pythonProject2/加入数据采样/f/{projectname}-'+(str)(_)+'.h5'
        model.save_weights(model_filename_h5)

    return models


# 自定义负相关学习损失函数
def negative_correlation_loss(y_true, y_pred, ensemble_predictions, lambda_ncl):
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    corr_loss = 0.0
    if len(ensemble_predictions) > 0:
        for pred in ensemble_predictions:
            corr_loss += tf.reduce_mean(tf.square(pred - y_pred))
        corr_loss /= len(ensemble_predictions)
    return mse_loss + lambda_ncl * corr_loss


# 模型加载函数
def load_models(projectname, n_estimators):
    from tensorflow.keras.models import model_from_json
    models = []
    for i in range(n_estimators):
        model = model_from_json(open(f'D:/pycharm/pythonProject2/加入数据采样/f/{projectname}-{i}.json').read())
        model.load_weights(f'D:/pycharm/pythonProject2/加入数据采样/f/{projectname}-{i}.h5', by_name=True)
        models.append(model)
    return models


# 模型测试函数
def test(X_test, y_test, projectname, n_estimators):
    models = load_models(projectname, n_estimators)
    ensemble_predictions = []
    for model in models:
        predictions = model.predict(X_test)
        ensemble_predictions.append(predictions)

    negative_correlation_predictions = []
    negative_correlation_probabilities = []
    for i in range(len(X_test)):
        ensemble_prediction = [predictions[i] for predictions in ensemble_predictions]
        negative_correlation_prediction = np.mean(ensemble_prediction)
        negative_correlation_probabilities.append(negative_correlation_prediction)
        negative_correlation_predictions.append(negative_correlation_prediction > 0.5)

    accuracy = accuracy_score(y_test, negative_correlation_predictions)
    f1 = f1_score(y_test, negative_correlation_predictions, average='binary')
    precision = precision_score(y_test, negative_correlation_predictions, average='binary')
    recall = recall_score(y_test, negative_correlation_predictions, average='binary')
    auc = roc_auc_score(y_test, negative_correlation_probabilities)

    return accuracy, f1, precision, recall, auc


# 十折交叉验证函数
def cross_validate(data, labels, n_estimators, lambda_ncl, projectname, num_folds=10):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "f1": [], "precision": [], "recall": [], "auc": []}

    fold = 1
    for train_index, test_index in kf.split(data):
        # print(f"Fold {fold}/{num_folds}")
        fold += 1

        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        models = train(X_train, y_train, n_estimators, lambda_ncl, projectname, X_train.shape[1])
        accuracy, f1, precision, recall, auc = test(X_test, y_test, projectname, n_estimators)

        metrics["accuracy"].append(accuracy)
        metrics["f1"].append(f1)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["auc"].append(auc)

    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}

    return avg_metrics





