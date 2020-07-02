import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

SL_DIGITS_DATASET = 'dataset/sign_language_digits_dataset/'
SL_MINIST = 'dataset/sign_language_mnist/'
ASL_ALPHABET = 'dataset/asl_alphabet/'


class Dataset:
    def __init__(self):
        self.digits = self._load_sl_digits_dataset()
        self.mnist = self._load_sl_mnist()

    def _load_sl_digits_dataset(self):
        labels = [9, 0, 7, 6, 1, 8, 4, 3, 2, 5]
        digits_x = np.load(SL_DIGITS_DATASET+'X.npy')
        raw_digits_y = np.load(SL_DIGITS_DATASET+'Y.npy')
        digits_y = np.array([], dtype=np.int)
        for label in raw_digits_y:
            digits_y = np.append(
                digits_y, labels[np.where(label == 1.0)[0][0]])
        return digits_x, digits_y

    def _load_sl_mnist(self):
        ds = np.genfromtxt(
            SL_MINIST+'sign_mnist_test.csv',
            delimiter=',',
            dtype=np.int,
            skip_header=1
        )
        [mnist_y, mnist_x] = np.split(ds, [1], axis=1)
        mnist_x, mnist_y = np.squeeze(mnist_x), np.squeeze(mnist_y)
        (num_imgs, _) = mnist_x.shape
        mnist_x = np.reshape(mnist_x, (num_imgs, 28, 28))/255
        return mnist_x, mnist_y

    def _load_asl_alphabet(self):
        pass

    def save_img(self, img, filename, label):
        pass

    def show(self):
        for i in range(10):
            img = self.mnist[0][i]
            label = self.mnist[1][i]
            plt.imshow(img)
            print(label)
            plt.show()
