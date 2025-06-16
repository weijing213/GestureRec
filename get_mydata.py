import cv2
import os
import numpy as np

# 获取用户输入的类别
category = input("请输入类别（例如：paper）：")

# 构建保存图像的文件夹路径
save_folder = os.path.join("mydata2", category)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 初始化图像计数器
count = 0

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

    # 将 ROI 从 BGR 转换为 BGRA 格式
    roi_rgba = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)

    # 在原始帧图像上绘制 300x300 区域的矩形框
    frame_with_rect = frame.copy()
    cv2.rectangle(frame_with_rect, (x, y), (x + 300, y + 300), (0, 255, 0), 2)

    # 显示包含矩形框的原始帧图像
    cv2.imshow('Camera Feed', frame_with_rect)

    # 显示提取的 300x300 区域图像（RGBA 格式）
    cv2.imshow('ROI', roi_rgba)

    key = cv2.waitKey(1)

    # 按下 's' 键保存图像
    if key & 0xFF == ord('s'):
        # 构建保存图像的文件名
        image_name = os.path.join(save_folder, f"{count}.png")
        # 将提取的 300x300 区域图像（RGBA 格式）保存为 PNG 格式
        cv2.imwrite(image_name, roi_rgba)
        count += 1
        print(f"已保存图像: {image_name}")

    # 按下 'q' 键退出程序
    elif key & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()

# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()
