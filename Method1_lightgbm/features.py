import cv2
import numpy as np
from skimage.feature import hog
def preprocess_image(image):
    """
    对输入图像进行预处理，包括调整大小、灰度化、自适应二值化和边缘检测。
    :param image: 输入的图像
    :return: 预处理后的图像
    """
    try:
        image = cv2.resize(image, (300, 300))
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
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
        masked_img = cv2.bitwise_and(image, image, mask=combined_binary)
        # 将掩码为 0 的区域设为白色
        masked_img[combined_binary == 0] = [255, 255, 255]
        edges = cv2.Canny(masked_img, 150, 300)
        return edges, masked_img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None
def convert_to_hog(image):
    """
    将输入图像转换为 HOG 特征。
    :param image: 输入的图像
    :return: HOG 特征列表
    """
    if image is not None:
        try:
            hog_array, hog_image = hog(image, orientations=8,
                                       pixels_per_cell=(20, 20),
                                       cells_per_block=(1, 1),
                                       visualize=True)
            return hog_array.tolist()
        except Exception as e:
            print(f"Error extracting HOG features: {e}")
    return None
def calculate_area(image):
    """
    计算图像中非零像素的面积
    :param image: 输入的图像
    :return: 面积
    """
    if len(image.shape) == 3:
        # 如果是三通道图像，转换为单通道
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray[gray == 255] = 0
    if gray is not None:
        return cv2.countNonZero(gray)
    return 0
def detect_corners(image):
    """
    检测图像中的角点位置
    :param image: 输入的图像
    :return: 角点坐标列表和标记角点后的图像
    """
    if image is not None:
        # 检测角点（限制最多检测10个角点）
        corners = cv2.goodFeaturesToTrack(image, 10, 0.15, 20)
        if corners is not None:
            corners = np.intp(corners)
            h, w = image.shape[:2]
            normalized_corners = []  # 归一化后的坐标列表
            for i in corners:
                x, y = i.ravel()
                normalized_corners.append((x / w, y / h))  # 归一化到 [0, 1]
            # 不足10个角点时补零
            while len(normalized_corners) < 10:
                normalized_corners.append((0, 0))
            return normalized_corners
    return [(0, 0)] * 10
def extract_features(image):
    """
    提取 HOG 特征、面积和角点坐标
    :param image: 输入的图像
    :return: 组合特征列表
    """
    edges, masked_img= preprocess_image(image)
    hog_features = convert_to_hog(edges)
    area = calculate_area(masked_img)
    corner_positions = detect_corners(edges)
    # 将角点坐标展平为一维列表
    corner_features = [coord for point in corner_positions for coord in point]
    if hog_features is not None:
        return hog_features + [area] + corner_features
    return None