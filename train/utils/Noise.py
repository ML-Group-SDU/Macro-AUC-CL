import random
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=1):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            # img = np.array(img / 255, dtype=float)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img = np.clip(img, 0, 255)
            # img = np.clip(img, 0.0, 1.0)
            # img = np.uint8(img*255)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img


# 椒盐噪声
class AddPepperNoise(object):
    """"
    Args:
        snr (float): Signal Noise Rate could be 0.99
        p (float): 概率值， 依概率执行
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  # 按概率进行
            # 把img转化成ndarry的形式
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            # 原始图像的概率（这里为0.9）
            signal_pct = self.snr
            # 噪声概率共0.1
            noise_pct = (1 - self.snr)
            # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            # 将mask按列复制c遍
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')  # 转化为PIL的形式
        else:
            return img


if __name__ == '__main__':
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        AddGaussianNoise(mean=random.uniform(0.5, 1.5), variance=0.5, amplitude=random.uniform(0, 45), p=0.5),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 测试查看
    img = Image.open("/home/zhangyan/data/Datasets/coco2017/validation/data/000000000139.jpg")
    img1 = AddGaussianNoise(mean=0, variance=1, amplitude=255)(img)
    # img1 = AddGaussianNoise()(img)
    plt.imshow(img1)
    plt.show()
    # img1.show()

    # img2 = AddPepperNoise(0.99, 1.0)(img1)
    # img2.show()
