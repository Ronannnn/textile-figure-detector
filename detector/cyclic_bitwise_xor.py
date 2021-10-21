import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from img_handle import ImgHandle


class CyclicBitwiseXor:
    def __init__(self, filename):
        self.img_handle = ImgHandle(filename, enhance_contrast=False, filter_high_pass=False)
        self.raw_img = self.img_handle.raw_img
        self.h, self.w, _ = self.raw_img.shape
        self.gray_img = self.img_handle.img
        self.img_handle.save(self.gray_img, "gray")

        # built-in parameters
        self.edge_thickness = 1
        self.circle_thickness = 2

    def calculate(self):
        avg_xor_dict = self.__xor_img(self.gray_img, self.w, 1)
        key_with_min_avg = min(avg_xor_dict.keys(), key=(lambda k: avg_xor_dict[k]))
        min_avg = avg_xor_dict[key_with_min_avg]
        points_y = [idx for idx, avg in avg_xor_dict.items() if avg <= min_avg * 1.7]
        circle_points = []
        for y in points_y:
            circle_points.append([int(self.h / 2), y])
        self.__draw_circles(circle_points)
        plt.plot(list(avg_xor_dict.keys()), list(avg_xor_dict.values()))
        plt.show()

    @staticmethod
    def __xor_img(raw_img, border, axis):
        img = raw_img.copy()
        avg_xor_dict = {}
        for i in range(1, border):
            rolled_img = np.roll(img, i, axis=axis)
            xor = cv.bitwise_xor(rolled_img, raw_img)
            avg_xor_dict[i] = np.average(xor)
        return avg_xor_dict

    def __draw_circles(self, points, color=(0, 0, 255)):
        new_img = self.raw_img.copy()
        for point in points:
            new_img = cv.circle(new_img, (point[1], point[0]), radius=0, color=color, thickness=self.circle_thickness)
        self.img_handle.save(new_img, "with-circles")


if __name__ == '__main__':
    CyclicBitwiseXor("../img/lattice/7.png").calculate()
