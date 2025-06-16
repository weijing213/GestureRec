import os
from shutil import copy
import random

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# 获取指定路径下的所有子文件夹（即类别名）
file_path = 'data'
flower_class = [item for item in os.listdir(file_path)
                if os.path.isdir(os.path.join(file_path, item))]

# 创建训练集目录及类别子目录
mkfile('shoushidata/train')
for cla in flower_class:
    mkfile(f'shoushidata/train/{cla}')

# 创建验证集目录及类别子目录
mkfile('shoushidata/test')
for cla in flower_class:
    mkfile(f'shoushidata/test/{cla}')
# 划分比例，训练集 : 验证集 = 9 : 1
split_rate = 0.2
# 遍历各类别图像并按比例划分
for cla in flower_class:
    cla_path = os.path.join(file_path, cla)
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))

    for index, image in enumerate(images):
        src_path = os.path.join(cla_path, image)
        if image in eval_index:
            dst_path = f'shoushidata/test/{cla}'
        else:
            dst_path = f'shoushidata/train/{cla}'

        copy(src_path, dst_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")

    print()

print("processing done!")