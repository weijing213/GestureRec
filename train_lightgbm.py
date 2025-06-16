import os
import numpy as np
import lightgbm as lgb
import cv2
from sklearn.metrics import accuracy_score, classification_report
import joblib  # 用于保存模型和数据
from Method1_lightgbm.features import extract_features
from tqdm import tqdm  # 进度条
import pandas as pd
# 假设你已经有了训练集和验证集的文件夹路径
train_dir = r'shoushidata/train'  # 训练集文件夹路径
val_dir = r'shoushidata/test'  # 验证集文件夹路径
# 定义类别标签
class_labels = ['paper', 'rock', 'scissors']  # 根据你的类别名称修改
# 数据增强函数
def augment_image(image):
    """
    对输入图像进行数据增强。
    :param image: 输入图像
    :return: 增强后的图像列表
    """
    augmented_images = []
    # 原始图像
    augmented_images.append(image)
    # 水平翻转
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    # 缩放
    height, width = image.shape[:2]
    scaled = cv2.resize(image, (int(width * 1.2), int(height * 1.2)))
    scaled = scaled[:height, :width]  # 裁剪回原始大小
    augmented_images.append(scaled)
    return augmented_images

# 加载数据集并提取特征
def load_data(data_dir, save_path_features, save_path_labels):
    if os.path.exists(save_path_features) and os.path.exists(save_path_labels):
        print(f"Loading pre - extracted features from {save_path_features}...")
        features = joblib.load(save_path_features)
        labels = joblib.load(save_path_labels)
    else:
        features = []
        labels = []
        for label in class_labels:
            label_path = os.path.join(data_dir, label)
            # 获取当前类别文件夹中的所有图像文件名
            img_names = os.listdir(label_path)
            # 使用 tqdm 包裹循环，添加进度条
            for img_name in tqdm(img_names, desc=f'Processing {label}'):
                img_path = os.path.join(label_path, img_name)
                image = cv2.imread(img_path)
                if image is not None:
                    # 数据增强
                    augmented_images = augment_image(image)
                    for aug_img in augmented_images:
                        feature = extract_features(aug_img)
                        if feature is not None:
                            features.append(feature)
                            labels.append(class_labels.index(label))
        features = np.array(features)
        labels = np.array(labels)
        joblib.dump(features, save_path_features)
        joblib.dump(labels, save_path_labels)
        print(f"Features saved to {save_path_features} and labels saved to {save_path_labels}")
    return features, labels

# 保存特征和标签的路径
train_features_path = 'Method1_lightgbm/checkpoints/train_features.joblib'
train_labels_path = 'Method1_lightgbm/checkpoints/train_labels.joblib'
val_features_path = 'Method1_lightgbm/checkpoints/val_features.joblib'
val_labels_path = 'Method1_lightgbm/checkpoints/val_labels.joblib'
# 加载训练集和验证集
print("Loading training data...")
X_train, y_train = load_data(train_dir, train_features_path, train_labels_path)
print("Loading validation data...")
X_val, y_val = load_data(val_dir, val_features_path, val_labels_path)
X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
X_val_df = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])])
# 定义LightGBM分类器的固定参数
params = {
    'learning_rate': 0.1,
    'n_estimators': 100,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multiclass',
    'num_class': len(class_labels),
    'verbose': -1
}
# 创建LightGBM分类器
lgb_clf = lgb.LGBMClassifier(
    **params,
)
# 训练模型
print("Training the LightGBM model...")
lgb_clf.fit(X_train_df, y_train)
# 保存模型权重文件
model_path = 'Method1_lightgbm/checkpoints/lgb_model_weights.joblib'
joblib.dump(lgb_clf, model_path)
print(f"模型权重已保存到: {model_path}")
# 验证集评估
print("Evaluating on validation set...")
val_pred = lgb_clf.predict(X_val_df)
val_accuracy = accuracy_score(y_val, val_pred)
print("验证集准确率:", val_accuracy)
print("验证集分类报告:")
print(classification_report(y_val, val_pred, target_names=class_labels))
