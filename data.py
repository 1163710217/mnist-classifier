# coding=utf-8

"""MNIST数据处理"""

import os
import urllib3


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

        # 数据下载链接
        self.urls = [
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
        ]

    def download(self):
        """下载数据"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        for url in self.urls:
            fpath = self.data_dir + url.split('/')[-1]
            if os.path.exists(fpath):
                continue

            print(fpath, end="...")
            http = urllib3.PoolManager()
            response = http.request('GET', url)
            with open(fpath, 'wb') as f:
                f.write(response.data)
            response.release_conn()
            print("完成")


def test():
    data = Mnist('./cache/')
    data.download()


if __name__ == '__main__':
    test()
