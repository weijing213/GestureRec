import os
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
class MyDataset(Dataset):
    def __init__(self, type, img_size, data_dir):
        self.img_size = img_size
        self.data_list = []  # 存储图片路径
        self.label_list = []  # 存储对应标签
        # 遍历数据目录，获取所有类别文件夹
        class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        self.class_to_label = {cls_name: idx for idx, cls_name in enumerate(class_folders)}  # 类别名到标签的映射
        # 遍历每个类别文件夹，收集图片路径和标签
        for class_name in class_folders:
            class_dir = os.path.join(data_dir, class_name)
            for file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, file)
                self.data_list.append(img_path)
                self.label_list.append(self.class_to_label[class_name])
        print(f"Load {type} Data Successfully! Total images: {len(self.data_list)}")
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img_path = self.data_list[item]
        label = self.label_list[item]
        img = Image.open(img_path)
        # 强制转换为RGB（无论原始是灰度图还是RGBA，都转成RGB）
        img = img.convert("RGB")  # 确保3通道
        img = img.resize((self.img_size, self.img_size))
        img = ToTensor()(img)
        label = tensor(label)
        return img, label