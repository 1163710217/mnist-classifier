# coding=utf-8

"""MNIST数据处理"""

import os
import gzip
import struct
import urllib3
import numpy as np


# 数据下载链接
urls = [
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
]


def download(url, save_path):
    """ 下载MNIST数据并保存到本地
    Args:
        url (str): 下载链接
        save_path (str): 保存路径
    """
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    with open(save_path, 'wb') as f:
        f.write(response.data)
    response.release_conn()


def uncompress(gz_file):
    """ 解压缩gz文件
    Args:
        gz_file (str): 压缩文件路径
    Returns:
        bytes: 解压后的二进制内容
    """
    with gzip.GzipFile(gz_file) as f:
        return f.read()


def parse_images(file_bytes):
    """ 解析图片集
    Args:
        file_bytes (bytes): 解压后的二进制数据
    Returns:
        list[np.ndarray]: 图片列表，每张图片为28×28的矩阵
    """
    offset = 4   # 跳过前4个字节（前4字节为校验值）
    img_cnt, rows, cols = struct.unpack_from('>III', file_bytes, offset)

    # 图片内容从第16个字节开始
    offset = 16
    imgs = []
    pixels = rows * cols
    format_str = f">{pixels}B"   # 每个像素占1个字节
    for i in range(img_cnt):
        # 读取图片像素
        img = struct.unpack_from(format_str, file_bytes, offset)
        offset += pixels

        # 将图片转换为28*28矩阵
        img = np.array(img).reshape((rows, cols))
        imgs.append(img)

    return imgs


def parse_labels(file_bytes):
    """ 解析标签集
    Args:
        file_bytes (bytes): 解压后的二进制数据
    Returns:
        list[int]: 标签列表，标签值为0~9
    """
    offset = 4
    label_cnt = struct.unpack_from('>I', file_bytes, offset)[0]

    offset = 8
    labels = []
    for i in range(label_cnt):
        label = int(struct.unpack_from(">B", file_bytes, offset)[0])
        offset += 1
        labels.append(label)

    return labels


class Mnist(object):
    def __init__(self, data_dir):
        """处理MNIST数据，提供样本和标签
        Args:
            data_dir (str): 数据存放目录
        """
        # 数据本地存放目录
        if data_dir[-1] != '/':
            data_dir += '/'
        self.data_dir = data_dir

        # 训练图片集
        self.train_images = None
        # 训练标签集
        self.train_labels = None
        # 测试图片集
        self.test_images = None
        # 测试标签集
        self.test_labels = None

    def download(self):
        """下载数据"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        for url in urls:
            fpath = self.data_dir + url.split('/')[-1]
            if os.path.exists(fpath):
                continue

            print(fpath, end="...")
            download(url, fpath)
            print("完成")

    def parse(self):
        """解析下载的数据"""
        self.download()

        train_images_path = self.data_dir + "train-images-idx3-ubyte.gz"
        train_labels_path = self.data_dir + "train-labels-idx1-ubyte.gz"
        test_images_path = self.data_dir + "t10k-images-idx3-ubyte.gz"
        test_labels_path = self.data_dir + "t10k-labels-idx1-ubyte.gz"

        self.train_images = parse_images(uncompress(train_images_path))
        self.train_labels = parse_labels(uncompress(train_labels_path))
        self.test_images = parse_images(uncompress(test_images_path))
        self.test_labels = parse_labels(uncompress(test_labels_path))


def test():
    data = Mnist('./cache/')
    data.download()
    data.parse()
    print(len(data.train_images))


if __name__ == '__main__':
    test()
