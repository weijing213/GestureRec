import cv2
import numpy as np
import time
from Method1_lightgbm.inference import inference
# 直接指定训练好的模型路径
model_filename = "Method1_lightgbm/checkpoints/lgb_model_weights.joblib"
# 打开摄像头
cap = cv2.VideoCapture(0)
# 初始化上次推理时间
last_inference_time = time.time()
result = None
while True:
    # 读取摄像头的一帧图像
    ret, frame = cap.read()
    if not ret:
        print("无法读取图像帧，请检查摄像头连接。")
        break
    # 对图像进行水平镜像翻转
    frame = cv2.flip(frame, 1)
    # 获取当前帧图像的高度和宽度
    height, width, _ = frame.shape
    # 计算 300x300 区域的左上角坐标，使其位于画面右上角
    x = width - 300
    y = 0
    # 从帧图像中提取 300x300 的区域
    roi = frame[y:y + 300, x:x + 300]
    # 在原始帧图像上绘制 300x300 区域的矩形框
    frame_with_rect = frame.copy()
    cv2.rectangle(frame_with_rect, (x, y), (x + 300, y + 300), (0, 255, 0), 2)
    # 转换到HSV颜色空间
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    # 转换到 HSV 颜色空间
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    # 分别提取 H、S、V 通道
    h_channel = hsv_img[:, :, 0]
    s_channel = hsv_img[:, :, 1]
    # 对 H、S、V 三个通道分别进行 OTSU 阈值分割
    _, h_binary = cv2.threshold(h_channel, 20, 255, cv2.THRESH_OTSU)
    _, s_binary = cv2.threshold(s_channel, 20, 255, cv2.THRESH_OTSU)
    # 合并三个二值图像
    combined_binary = cv2.bitwise_and(h_binary, s_binary)
    # 对合并后的二值图像进行去噪操作
    # 开运算
    kernel = np.ones((3, 3), np.uint8)
    combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel, iterations=3)
    # 中值滤波
    combined_binary = cv2.medianBlur(combined_binary, 7)
    # 应用掩码到原始图像
    masked_img = cv2.bitwise_and(roi, roi, mask=combined_binary)
    # 将掩码为 0 的区域设为白色
    masked_img[combined_binary == 0] = [255, 255, 255]
    # 检查是否经过了1秒
    current_time = time.time()
    if current_time - last_inference_time >= 1:
        result = inference(roi, model_filename)
        last_inference_time = current_time
    if result is not None:
        cv2.putText(frame_with_rect, f"Class: {result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # 显示包含矩形框和检测结果的原始帧图像
    cv2.imshow('Camera Feed', frame_with_rect)
    # 显示Canny分割后的图像
    cv2.imshow('blurred', masked_img)
    key = cv2.waitKey(1)
    # 按下 'q' 键退出程序
    if key & 0xFF == ord('q'):
        break
# 释放摄像头资源
cap.release()
# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()
