# coding=utf-8

"""最近邻分类器"""


import numpy as np
from sklearn.metrics import confusion_matrix


class NearestNeighbor(object):
    def __init__(self):
        """ 最近邻模型
            为方便计算 # 将每张28*28的二维图片转为长度为748的一维向量，
            用二维矩阵存放所有训练集图片，维度为(N, 748)，N为图片数量。
            用一维向量存放训练集标签，维度为(N,)。
        """
        self.train_images = None
        self.train_labels = None


def train(train_images, train_labels):
    """ 训练模型
    Args:
        train_images (list[np.ndarray]): 图片集
        train_labels (list[int]): 标签集
    Returns:
        NearestNeighbor: 训练好的模型
    """
    model = NearestNeighbor()
    model.train_images = np.array([image.flatten() for image in train_images])
    model.train_labels = np.array(train_labels)
    return model


def predict(model, test_image):
    """ 预测图片类别
    Args:
        model (NearestNeighbor): 训练好的模型
        test_image (np.ndarray): 测试图片，28×28矩阵
    Returns:
        int: 预测的类别（0~9）
    """
    # 将测试图片转换为一维向量，以便计算
    test_image = test_image.flatten()
    all_l1 = np.sum(np.abs(model.train_images - test_image), axis=1)
    similar_image_index = np.argmin(all_l1)
    label = model.train_labels[similar_image_index]
    return label


def get_confusion_matrix(model, test_images, test_labels):
    """ 用测试集评估模型，得到混淆矩阵
    Args:
        model (NearestNeighbor): 训练好的模型
        test_images (list[np.ndarray]): 测试集图片
        test_labels (list[int]): 测试集标签
    Returns:
        np.array: 混淆矩阵
    """
    pred_labels = []
    for i, image in enumerate(test_images):
        pred_labels.append(predict(model, image))

        if (i+1) % 100 == 0:
            print(i)

    return confusion_matrix(test_labels, pred_labels)


def test():
    from data import Mnist
    mnist = Mnist("./cache/")
    model = train(mnist.train_images, mnist.train_labels)
    confmat = get_confusion_matrix(model, mnist.test_images, mnist.test_labels[:500])
    print(confmat)


if __name__ == '__main__':
    test()
