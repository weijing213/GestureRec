import torch
import cv2
from torchvision.transforms import ToTensor
from Method2_vgg import vgg_model
class_names = ["paper", "rock", "scissors"]  # 根据你的数据集类别修改
def real_time_detection(model_path, img_size=128, num_class=3, gpu=True):
    # 初始化模型
    if gpu:
        net = vgg_model.VGG(img_size=img_size, input_channel=3, num_class=num_class).cuda()
    else:
        net = vgg_model.VGG(img_size=img_size, input_channel=3, num_class=num_class)
    # 加载模型参数
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break

        # 镜像显示
        frame = cv2.flip(frame, 1)

        # 获取右上角 300x300 的 ROI 区域
        height, width, _ = frame.shape
        roi_x = width - 300
        roi_y = 0
        roi = frame[roi_y:roi_y + 300, roi_x:roi_x + 300]

        # 绘制 ROI 区域的矩形框
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + 300, roi_y + 300), (0, 255, 0), 2)

        # 图像预处理
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = ToTensor()(img)
        img = img.unsqueeze(0)  # 添加批次维度
        if gpu:
            img = img.cuda()

        # 推理
        with torch.no_grad():
            output = net(img)
            _, indices = torch.max(output, 1)
            percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
            perc = percentage[int(indices)].item()
            result = class_names[indices.item()]  # 转换为Python标量

        # 在帧上显示结果
        cv2.putText(frame, f'Predicted: {result}, Confidence: {perc:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time Detection', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 定义参数
    model_path = "Method2_vgg/checkpoints/epoch_10-best-acc_1.0.pth"
    img_size = 128
    num_class = 3
    gpu = True

    real_time_detection(model_path, img_size, num_class, gpu)