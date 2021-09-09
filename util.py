import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg16


def preprocess_image(image_path, width, height):
    """图像预处理

    :param image_path: 目标图像路径
    :param width: 目标图像宽度
    :param height: 目标图像高度
    :return: tensor
    """

    # 读取图像并resize
    img = load_img(image_path, target_size=(height, width))

    # 数据转换至 Numpy array
    img = img_to_array(img)

    # 增加维度 使输入数据符合模型预期
    # (1, height, width, channel)
    img = np.expand_dims(img, axis=0)

    # 模型输入数据预处理, 归一化等
    img = vgg16.preprocess_input(img)

    return img


def deprocess_image(x):
    """图像后处理

    将模型输出转换为可存储数据

    :param x:
    :return:
    """

    # 常规图像格式解析各通道像素点值应为0~255,
    # 但由于模型处理时将其中心化了(-128~127),
    # 因此需要去中心化
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # BGR2RGB
    x = np.clip(x, 0, 255).astype('uint8')  # 浮点取整, 舍去小数
    return x


def enprocess_image(x):
    """数据预处理 (模型)

    将图像数据转换为模型可以读取的数据

    :param x:
    :return:
    """
    x[:, :, 0] -= 103.939  # zero-centering 0中心化：减去均值
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    x = x[:, :, ::-1]  # RGB2BGR
    return x
