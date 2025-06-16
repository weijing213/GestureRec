import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from Method2_vgg import vgg_model
class_names = ["paper", "rock", "scissors"]  # 根据你的数据集类别修改
def image_classify_folder(folder_path, model_path, img_size=128, num_class=3, gpu=True):
    # 初始化模型
    if gpu:
        net = vgg_model.VGG(img_size=img_size, input_channel=3, num_class=num_class).cuda()
    else:
        net = vgg_model.VGG(img_size=img_size, input_channel=3, num_class=num_class)
    # 加载模型参数
    net.load_state_dict(torch.load(model_path))
    net.eval()
    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            try:
                # 图像预处理
                img = Image.open(img_path)
                img = img.convert("RGB")  # 确保RGB格式
                img = img.resize((img_size, img_size))
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
                    print(f'Image: {filename}, Predicted: {result}, Confidence: {perc:.2f}%')
            except Exception as e:
                print(f"Error processing image {filename}: {e}")

if __name__ == "__main__":
    # 定义参数
    folder_path = "finaltest"  # 文件夹路径
    model_path = "Method2_vgg/checkpoints/epoch_10-best-acc_1.0.pth"
    img_size = 128
    num_class = 3
    gpu = True

    image_classify_folder(folder_path, model_path, img_size, num_class, gpu)