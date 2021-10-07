import time

import cv2 as cv
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

from stripe_detector import StripeDetector


class CyclicBitwiseXor:
    def __init__(self, filename):
        # filename info
        filepath = Path(filename)
        self.parent_dir = filepath.parent.absolute()
        self.fn_stem = filepath.stem  # filename without parent dir and suffix
        self.fn_suffix = filepath.suffix  # suffix with dot, e.g. ".jpg"

        # img preprocess
        self.raw_img = cv.imread(filename)
        self.h, self.w, _ = self.raw_img.shape
        self.size = self.h * self.w
        self.weaken_shadow_img = StripeDetector.weaken_shadow(self.raw_img)
        self.gray_scaled_img = cv.cvtColor(self.weaken_shadow_img, cv.COLOR_RGB2GRAY)
        self.__show_and_save(self.gray_scaled_img, "test", show=True, save=False)

        # built-in parameters
        self.edge_thickness = 1
        self.circle_thickness = 2

    def calculate(self):
        avg_xor_dict = self.__xor_img(self.gray_scaled_img, self.w, 1)
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
        self.__show_and_save(new_img, "with-circles")

    def __show_and_save(self, img, label, show=False, save=True):
        cv.imshow(label, img) if show else None
        cv.imwrite(os.path.join(self.parent_dir, self.fn_stem + "[" + label + "]" + self.fn_suffix), img) if save else None


if __name__ == '__main__':
    CyclicBitwiseXor("img/lattice/7.png").calculate()
