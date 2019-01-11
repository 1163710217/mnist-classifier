# coding=utf-8

"""最近邻分类器"""


import numpy as np
from sklearn.metrics import confusion_matrix


def l1(img1, img2):
    """ 计算图片间的L1距离
    Args:
        img1 (np.ndarray): 二维数组（图片像素为28×28）
        img2 (np.ndarray): 二维数组
    Returns:
        float: 图片L1距离
    """
    return np.sum(abs(img1 - img2))


class NearestNeighbor(object):
    def __init__(self):
        """ 最近邻模型 """
        self.train_images = []
        self.train_labels = []


def train(train_images, train_labels):
    """ 训练模型
    Args:
        train_images (list[np.ndarray]): 图片集
        train_labels (list[int]): 标签集
    Returns:
        NearestNeighbor: 训练好的模型
    """
    model = NearestNeighbor()
    model.train_images = train_images
    model.train_labels = train_labels
    return model


def predict(model, test_image):
    """ 预测图片类别
    Args:
        model (NearestNeighbor): 训练好的模型
        test_image (np.ndarray): 测试图片，28×28矩阵
    Returns:
        int: 预测的类别（0~9）
    """
    all_l1 = [l1(test_image, image) for image in model.train_images]
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
    confmat = get_confusion_matrix(model, mnist.test_images[:500], mnist.test_labels[:500])
    print(confmat)


if __name__ == '__main__':
    test()
