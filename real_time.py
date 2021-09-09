import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import load_img
from scipy.optimize import fmin_l_bfgs_b
from imageio import imsave

from util import preprocess_image, deprocess_image
from loss import total_loss

ITERATIONS = 30  # 迭代次数, 越大越趋近于理想混合图片
CONTENT_IMAGE_PATH = "content.jpg"  # 内容图像路径
STYLE_IMAGE_PATH = "style.jpg"  # 风格图像路径

WIDTH, HEIGHT = load_img(CONTENT_IMAGE_PATH).size
INPUT_HEIGHT = 400
INPUT_WIDTH = int(WIDTH * INPUT_HEIGHT / HEIGHT)  # 通过内容图像尺寸计算预处理模型 INPUT_SIZE 时所需参数

CONTENT_WEIGHT = 0.3  # 超参α
STYLE_WEIGHT = 1.  # 超参β (层风格占比=β/风格层总数)
# 此处若选择 β = 1. , α = 0.025 即实际输出时的风格与内容比率为 (1.0 / 5) : 0.025 = 0.25 : 0.025 = 10 : 1


class Evaluator:
    """评估器类

    """

    def __init__(self):
        self.loss_value = None  # 损失值缓冲
        self.grads_values = None  # 梯度值缓冲

    def loss(self, x):
        """损失函数

        :param x:
        :return:
        """

        # 数据检查断言
        assert self.loss_value is None

        # 将1维向量还原为3维tensor
        x = x.reshape((1, INPUT_HEIGHT, INPUT_WIDTH, 3))

        # 计算反向传播时权重的更新数值(计算梯度\求总loss关于生成图特征的导数)
        outs = fetch_loss_and_grads([x])

        # 得到损失值
        self.loss_value = outs[0]

        # 得到梯度值
        self.grad_values = outs[1].flatten().astype('float64')

        return self.loss_value

    def grads(self, x):
        """梯度计算

        :param x:
        :return:
        """
        # 数据检查断言
        assert self.loss_value is not None

        # 从已有数据拷贝
        grad_values = np.copy(self.grad_values)

        # reset
        self.loss_value = None
        self.grad_values = None

        return grad_values


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()  # 关闭即时模式

    # 以CPU进行预测 (备注掉此段将优先使用GPU进行预测)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # GPU显存配置 (若OOM错误通常为显存不足, 此处对tensor进行分片再传入GPU处理)
    # config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    content_image = K.constant(preprocess_image(CONTENT_IMAGE_PATH, INPUT_WIDTH, INPUT_HEIGHT))
    style_image = K.constant(preprocess_image(STYLE_IMAGE_PATH, INPUT_WIDTH, INPUT_HEIGHT))
    generate_image = K.placeholder((1, INPUT_HEIGHT, INPUT_WIDTH, 3))  # 随机生成一张图片

    # 生成模型输入tensor
    # 此处各图像数据顺序会影响后续读出layer数据的顺序
    # 目前为 0=内容图像, 1=风格图像, 2=生成图像
    input_tensor = K.concatenate([content_image, style_image, generate_image], axis=0)

    # 创建模型
    model = vgg16.VGG16(
        input_tensor=input_tensor,
        weights="imagenet",
        include_top=False
    )

    # 生成各层数据键值映射
    # p: 内容
    # a: 风格
    # x: 生成
    p, a, x = {}, {}, {}
    for layer in model.layers:
        # 只响应卷积层, 因为我们只需要卷积层的特征
        if "conv" not in layer.name:
            continue
        p[layer.name] = layer.output[0, :, :, :]  # 内容
        a[layer.name] = layer.output[1, :, :, :]  # 风格
        x[layer.name] = layer.output[2, :, :, :]  # 生成

    # 计算各卷积特征的损失值
    loss = total_loss(p, a, x, STYLE_WEIGHT, CONTENT_WEIGHT)

    # 实例化梯度下降评估器
    evaluator = Evaluator()

    # 计算梯度, 即总loss值对生成图像的导数, 即反向传播时所需更新权重的幅度
    grads = K.gradients(loss, generate_image)[0]

    # 基于损失值和梯度值, 定义计算生成图数据
    fetch_loss_and_grads = K.function([generate_image], [loss, grads])

    # 生成初始数据
    x = preprocess_image(CONTENT_IMAGE_PATH, INPUT_WIDTH, INPUT_HEIGHT)
    x = x.flatten()  # 铺平tensor (转换为向量供L-BFGS算法使用)

    for i in range(ITERATIONS):
        print("开始第 {}/{} 轮迭代".format(i+1, ITERATIONS))
        start_time = time.time()

        # L-BFGS梯度下降
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                         fprime=evaluator.grads, maxfun=20)
        print('当前loss:', min_val)

        # 输出当前图片
        img = x.copy().reshape((INPUT_HEIGHT, INPUT_WIDTH, 3))  # 将1维向量还原为3维tensor并copy
        img = deprocess_image(img)  # 数据后处理
        file_name = './iteration{}_loss{}.png'.format(i, min_val)
        imsave(file_name, img)
        print('结果存储为: ', file_name)

        end_time = time.time()
        print('第 {} 轮迭代完成, 耗时 {} 秒'.format(i + 1, end_time - start_time))
