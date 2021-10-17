import os
from pathlib import Path
import cv2 as cv
import numpy as np


class SimDetector:
    def __init__(self, filename, l_x, l_y, r_x, r_y):
        # filename info
        filepath = Path(filename)
        self.parent_dir = filepath.parent.absolute()
        self.fn_stem = filepath.stem  # filename without parent dir and suffix
        self.fn_suffix = filepath.suffix  # suffix with dot, e.g. ".jpg"

        self.pt_h = r_x - l_x
        self.pt_w = r_y - l_y

        self.raw_img = cv.imread(filename)
        self.h, self.w, _ = self.raw_img.shape
        self.size = self.h * self.w
        self.gray_scaled_img = cv.cvtColor(self.raw_img, cv.COLOR_RGB2GRAY)
        self.template = self.gray_scaled_img[l_x: r_x, l_y: r_y]

        # img preprocess
        self.raw_img = cv.imread(filename)

        self.circle_thickness = 1

    def ncc_with_cv(self):
        res = cv.matchTemplate(self.gray_scaled_img, self.template, cv.TM_CCORR_NORMED)
        threshold = 0.999
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(self.raw_img, pt, (pt[0] + self.pt_w, pt[1] + self.pt_h), (0, 255, 255), 1)
        self.__show_and_save(self.raw_img, "test")

    def ncc_with_customized_fn(self):
        ncc_list = []
        for x in range(self.h - self.pt_h):
            for y in range(self.w - self.pt_w):
                ncc = self.ncc(
                    self.template,
                    self.gray_scaled_img[x: x + self.pt_h, y: y + self.pt_w]
                )
                ncc_list.append(ncc)
        print(ncc_list)

    @staticmethod
    def ncc(img1, img2):
        return np.mean(np.multiply((img1 - np.mean(img1)), (img2 - np.mean(img2)))) / (np.std(img1) * np.std(img2))

    def draw_edges(self, x, y, color=(0, 0, 255)):
        new_img = self.raw_img.copy()
        pts = []
        # top and bottom
        for i in range(y, y + self.pt_w + 1):
            pts.append((x, i))
            pts.append((x + self.pt_h, i))
        # left and right
        for i in range(x + 1, x + self.pt_h):
            pts.append((i, y))
            pts.append((i, y + self.pt_w))
        for p in pts:
            new_img = cv.circle(new_img, (p[1], p[0]), radius=0, color=color, thickness=self.circle_thickness)
        self.__show_and_save(new_img, "with-circles")

    def __show_and_save(self, img, label, show=True, save=True):
        cv.imshow(label, img) if show else None
        cv.imwrite(os.path.join(self.parent_dir, self.fn_stem + "[" + label + "]" + self.fn_suffix), img) if save else None

    # todo 基于描述符


if __name__ == '__main__':
    sd = SimDetector("img/lattice/p1.png", 45, 48, 90, 90)
    sd.ncc_with_cv()
