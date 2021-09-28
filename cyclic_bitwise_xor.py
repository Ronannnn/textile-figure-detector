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
        # key: xor sum, value: idx
        xor_dict = {}
        cloned_img = self.gray_scaled_img.copy()
        # for line chart
        line_x = []
        line_y = []
        for idx in range(self.w - 1):
            # left shift by one pixel each time
            cloned_img = np.roll(cloned_img, -1, axis=1)
            # xor and count
            key = (cloned_img != self.gray_scaled_img).sum()
            if key not in xor_dict:
                xor_dict[key] = []
            xor_dict[key].append(idx)
            line_x.append(idx)
            line_y.append(key)
        dict_keys = np.array(list(xor_dict.keys()))
        min_avg = np.min(dict_keys)
        min_keys = dict_keys[dict_keys <= min_avg * 1.0004]
        circle_points = []
        for key in min_keys:
            for w_idx in xor_dict[key]:
                circle_points.append([int(self.h / 2), w_idx])
        self.__draw_circles(circle_points)
        plt.plot(line_x, line_y)
        plt.show()

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
