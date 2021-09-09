from tensorflow.keras import backend as K

CONTENT_LAYER = "block5_conv2"  # VGG-Net中的conv5_2
STYLE_LAYERS = ["block{}_conv1".format(i) for i in range(1, 6)]  # VGG-Net中各卷积的底层(conv1_1, conv2_1, ...)


def content_loss(p, x):
    """内容特征损失函数

    :param p: 内容特征
    :param x: 生成图像
    :return: tensor
    """

    # 公式1中 F_ij^l
    f_ij = x[CONTENT_LAYER]

    # 公式1中 P_ij^l
    p_ij = p[CONTENT_LAYER]

    # 公式1
    return K.sum(K.square(f_ij - p_ij)) / 2.


def calc_gram(matrix):
    """计算目标矩阵的gram矩阵

    gram矩阵为目标矩阵中数据相互关系的映射

    :param matrix: 目标矩阵
    :return: tensor
    """

    # K.permute_dimensions 为矩阵翻转操作
    # 对于3维矩阵而言, 参数中 0, 1, 2 的位置分别代表转置后x, y, z轴位置
    # 此处选择参数 2, 0, 1 进行翻转操作后, 矩阵中x→y, y→z, z→x, 以此完成数据的乱序

    # K.batch_flatten 为铺平操作
    # 其会对于参数中的矩阵进行铺平, 最终返回一维的tensor
    i = K.batch_flatten(K.permute_dimensions(matrix, (2, 0, 1)))

    # 通过对矩阵 i 进行转置操作得到矩阵 j
    j = K.transpose(i)

    # i j 矩阵相乘得到 gram 矩阵
    # 公式2
    return K.dot(i, j)


def error_loss(a_l, x_l):
    """单层的风格特征损失

    :param a_l: 风格特征
    :param x_l: 生成图像
    :return: tensor
    """

    # 公式3中 G_ij^l
    g_ij = calc_gram(a_l)

    # 公式3中 A_ij^l
    a_ij = calc_gram(x_l)

    # tensor尺寸, tuple
    shape = K.int_shape(a_l)
    col = shape[0]  # tensor 列数
    row = shape[1]  # tensor 行数

    # 公式3中 N_l
    n = col * row

    # tensor深度, 即通道数
    # 公式3中 M_l
    m = len(K.int_shape(a_l))

    # 公式3
    return K.sum(K.square(g_ij - a_ij)) / (4. * (n ** 2) * (m ** 2))


def style_loss(a, x, w_l):
    """风格损失函数

    :param a: 风格特征
    :param x: 生成图像
    :param w_l: 各层权重, 公式4中 w_l
    :return: tensor
    """

    result = K.variable(0.)  # 实例化一个新的tensor, 用于存储运算结果
    for layers_name in a:
        if layers_name in STYLE_LAYERS:
            result = result + w_l * error_loss(a[layers_name], x[layers_name])

    return result


def total_loss(p, a, x, style_weight, content_weight):
    """最终损失函数

    :param content_weight: 内容权重
    :param style_weight: 风格权重
    :param p: 内容特征
    :param a: 风格特征
    :param x: 生成图像
    :return: tensor
    """

    # 通过超参β计算各Style层权重 w_l
    w_l = (style_weight / len(STYLE_LAYERS))

    # 公式5 (其中超参β在[style_loss]中参与运算)
    return content_weight * content_loss(p, x) + style_loss(a, x, w_l)
