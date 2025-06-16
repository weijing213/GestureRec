**项目结构**
```
    - Task1
        - finaltest
        - Method1
          - checkpoints
          - features.py
          - inference.py
        - Method2
          - checkpoints
          - vgg_dataset.py
          - vgg_model.py
        - mydata
        - shoushidata
          - test
          - train
        - get_mydata.py
        - process_data.py
        - seefeature.py
        - test_lightgbm.py
        - test_vgg.py
        - time_lightgbm.py
        - time_vgg.py
        - train_lightgbm.py
        - train_vgg.py
        - Readme.md
        - requirements.txt
```
**项目说明**
- 本项目为石头剪刀布手势识别，其中采用了传统特征提取和深度学习网络算法分别对手势进行识别，训练数据为3类手势图片数据，
测试数据为3类手势图片数据，训练之后测试输出图像类别。
- 方法一：采用HSV色彩分割，将手势ROI区域提取出来，之后提取完整手势边缘图，提取HOG、角点、面积特征、采用LightGBM进行训练
得出图像分类模型1
- 方法二：采用VGG19卷积网络，直接对3通道彩色图像进行处理，16个卷积层对其进行特征提取，3个全连接层完成图像分类
- 此外，项目还可以进行手势图像的实时检测。

**项目环境**
```
    Python 3.11.11
    所需库：
        requirements.txt
```
**数据采集**
```
    Python get_mydata.py
    采用opencv进行图像采集，将图像进行裁剪，并保存为3类手势图片数据于mydata文件夹下
```
**数据文件夹**
```
    process_data.py可对一个形如
    mydata
        |--label1
        |--label2
        |--label3
    的数据文件夹快速划分测试集和训练集
    将获取到的数据集放入shoushidata文件夹下，文件结构为
    shoushidata
        |--test
            |--label1
            |--label2
            |--label3
        |--train
            |--label1
            |--label2
            |--label3
```
**训练_方法一**
```
    Python train_lightgbm.py
```
**训练_方法一**
```
    Python train_vgg.py
```
**权重文件**
```
    Method1
        |--checkpoints
            |--lgb_model_weights.joblib
    Method2
        |--checkpoints
            |--epoch_10-best-acc_1.0.pth

```
**测试_方法一**
```
    Python test_lightgbm.py
    Python time_lightgbm.py ##(实时检测)
```
**测试_方法二**
```
    Python test_vgg.py
    Python time_vgg.py ##(实时检测)
```