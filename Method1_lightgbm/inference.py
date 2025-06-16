import joblib
import pandas as pd
import numpy as np
from Method1_lightgbm.features import extract_features
# 定义类别标签
class_labels = ['paper', 'rock', 'scissors']
def inference(image, model_filename):
    """
    对输入图像进行推理并输出其相应类别名称。
    :param image: 输入图像数据（numpy数组）
    :param model_filename: 训练好的模型的文件名
    :param pca_filename: 训练好的PCA模型的文件名
    :return: 预测的类别名称
    """
    # 检查图像是否有效
    if image is None:
        print("Error: 输入图像无效")
        return None
    # 提取特征
    features = extract_features(image)
    if features is None:
        print("Error: 无法提取图像特征")
        return None
    # 将特征列表转换为 NumPy 数组
    features = np.array(features)
    # 检查特征数组是否为一维，如果是则将其转换为二维
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    # 将特征转换为 DataFrame 并添加列名
    n_features = features.shape[1]
    features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(n_features)])
    # 加载模型
    try:
        model = joblib.load(model_filename)
    except Exception as e:
        print(f"Error: 无法加载模型文件 {model_filename}. 错误信息: {e}")
        return None
    # 进行预测
    try:
        prediction = model.predict(features_df)
    except Exception as e:
        print(f"Error: 预测失败. 错误信息: {e}")
        return None
    # 将数字标签转换为类别名称
    predicted_class_name = class_labels[prediction[0]]
    return predicted_class_name