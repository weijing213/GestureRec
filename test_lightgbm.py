import os
from Method1_lightgbm.inference import inference
import cv2
if __name__ == "__main__":
    # 直接指定要预测的图像文件夹路径
    image_folder = "finaltest"
    # 直接指定训练好的模型路径
    model_filename = "Method1_lightgbm/checkpoints/lgb_model_weights.joblib"
    # 遍历文件夹中的所有图像
    for filename in os.listdir(image_folder):
        file_path = os.path.join(image_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)
            result = inference(image, model_filename)
            if result is not None:
                print(f"The predicted class of {filename} is: {result}")