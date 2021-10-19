import os
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
from sklearn.cluster import AgglomerativeClustering


class SimDetector:
    def __init__(self, filename, l_x, l_y, r_x, r_y, c_x, c_y):
        # filename info
        filepath = Path(filename)
        self.parent_dir = filepath.parent.absolute()
        self.fn_stem = filepath.stem  # filename without parent dir and suffix
        self.fn_suffix = filepath.suffix  # suffix with dot, e.g. ".jpg"

        self.pt_h = r_x - l_x
        self.pt_w = r_y - l_y
        self.c_h = c_x - l_x
        self.c_w = c_y - l_y

        self.raw_img = cv.imread(filename)
        self.h, self.w, _ = self.raw_img.shape
        self.size = self.h * self.w
        self.gray_scaled_img = cv.cvtColor(self.raw_img, cv.COLOR_RGB2GRAY)
        self.template = self.gray_scaled_img[l_x: r_x, l_y: r_y]

        self.circle_thickness = 1

    # Ref: https://www.geeksforgeeks.org/template-matching-using-opencv-in-python/
    # zip[*loc::-1] explanation: https://stackoverflow.com/questions/56449024/explanation-of-a-few-lines-template-matching-in-python-using-opencv
    def ncc_with_cv(self, ncc_thresh, dis_thresh):
        ncc = cv.matchTemplate(self.gray_scaled_img, self.template, cv.TM_CCORR_NORMED)
        loc = np.where(ncc >= ncc_thresh)
        raw_pts = [list(a) for a in zip(*loc)]
        print(len(raw_pts))
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=dis_thresh).fit_predict(raw_pts)
        cluster_dict = {}
        for idx, label in enumerate(labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(idx)
        pts = []
        for _, cluster in cluster_dict.items():
            min_ncc = sys.maxsize
            x, y = -1, -1
            for idx in cluster:
                x, y = raw_pts[idx][0], raw_pts[idx][1]
                if min_ncc > ncc[x][y]:
                    min_nnc = ncc[x][y]
            pts.append([y, x])  # x y are reverse in img
        img_rec = self.raw_img.copy()
        img_pt = self.raw_img.copy()
        for pt in pts:
            cv.rectangle(img_rec, pt, (pt[0] + self.pt_w, pt[1] + self.pt_h), (0, 255, 255), self.circle_thickness)
            img_pt = cv.circle(img_pt, (pt[0] + self.c_w, pt[1] + self.c_h), radius=0, color=(0, 255, 255), thickness=3)
        self.__show_and_save(img_rec, "rec")
        self.__show_and_save(img_pt, "pt")

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
        img_rec = self.raw_img.copy()
        img_pt = self.raw_img.copy()
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
            img_rec = cv.circle(img_rec, (p[1], p[0]), radius=0, color=color, thickness=self.circle_thickness)
        self.__show_and_save(img_rec, "with-circles")

    def __show_and_save(self, img, label, show=True, save=True):
        cv.imshow(label, img) if show else None
        cv.imwrite(os.path.join(self.parent_dir, self.fn_stem + "[" + label + "]" + self.fn_suffix), img) if save else None


if __name__ == '__main__':
    sd = SimDetector("img/lattice/p1.png", 45, 48, 90, 90, 67, 69)
    sd.ncc_with_cv(0.9988, 10)
    # sd = SimDetector("img/lattice/p3.png", 130, 90, 360, 285, 245, 187)
    # sd.draw_edges(130, 90)
    # sd.ncc_with_cv(0.9, 200)
