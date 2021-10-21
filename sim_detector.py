import os
import sys
from pathlib import Path
import cv2 as cv
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from skimage.feature import local_binary_pattern

from stripe_detector import StripeDetector
from img_util import ImgUtil


class SimDetector:
    def __init__(self, filename, l_x, l_y, r_x, r_y, c_x, c_y):
        self.filename = filename

        self.pt_h = r_x - l_x
        self.pt_w = r_y - l_y
        self.c_h = c_x - l_x
        self.c_w = c_y - l_y

        self.raw_img = cv.imread(filename)
        self.contrast_enhanced_img = ImgUtil.enhance_contrast(self.raw_img)
        self.gray_img = cv.cvtColor(self.contrast_enhanced_img, cv.COLOR_RGB2GRAY)
        self.gray_img = ImgUtil.filter_high_pass(self.gray_img, 2)
        self.template = self.gray_img[l_x: r_x, l_y: r_y]

        self.circle_thickness = 1

    # Ref: https://www.geeksforgeeks.org/template-matching-using-opencv-in-python/
    # zip[*loc::-1] explanation: https://stackoverflow.com/questions/56449024/explanation-of-a-few-lines-template-matching-in-python-using-opencv
    def ncc_with_cv(self, ncc_thresh, dis_thresh):
        self.match_template(self.gray_img, self.template, ncc_thresh, dis_thresh)

    def lbp(self, ncc_thresh, dis_thresh, p, r):
        gray_lbp = local_binary_pattern(self.gray_img, p, r, 'uniform')
        template_lbp = local_binary_pattern(self.template, p, r, 'uniform')
        gray_lbp = np.array(gray_lbp).astype(np.uint8)
        template_lbp = np.array(template_lbp).astype(np.uint8)
        self.match_template(gray_lbp, template_lbp, ncc_thresh, dis_thresh)

    def match_template(self, raw, template, ncc_thresh, dis_thresh):
        ncc = cv.matchTemplate(raw, template, cv.TM_CCOEFF_NORMED)
        print(ncc)
        loc = np.where(ncc >= ncc_thresh)
        pts = [list(a) for a in zip(*loc)]
        print(len(pts))
        pts = self.cluster(ncc, pts, dis_thresh)
        img_rec = self.raw_img.copy()
        # img_pt = self.raw_img.copy()
        for pt in pts:
            cv.rectangle(img_rec, (pt[1], pt[0]), (pt[1] + self.pt_w, pt[0] + self.pt_h), (0, 255, 255), self.circle_thickness)
            # img_pt = cv.circle(img_pt, (pt[1] + self.c_w, pt[0] + self.c_h), radius=0, color=(0, 255, 255), thickness=3)
        ImgUtil.save(img_rec, self.filename, "rec")
        # ImgUtil.save(img_pt, self.filename, "pt")

    @staticmethod
    def cluster(ncc, raw_pts, dis_thresh):
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=dis_thresh, compute_full_tree=True).fit_predict(raw_pts)
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
            pts.append([x, y])  # x y are reverse in img
        return pts

    def draw_edges(self, x, y, color=(0, 0, 255)):
        copied_img = self.raw_img.copy()
        img_rec = ImgUtil.draw_rectangle(copied_img, x, y, self.pt_h, self.pt_w, color, self.circle_thickness)
        ImgUtil.save(img_rec, self.filename, "template")


if __name__ == '__main__':
    sd = SimDetector("img/lattice/p1.png", 45, 48, 90, 90, 67, 69)
    # sd.ncc_with_cv(0.9988, 10)
    sd.lbp(0.1, 20, p=40, r=1)
    # sd = SimDetector("img/lattice/p3.png", 130, 90, 360, 285, 245, 187)
    # sd.ncc_with_cv(0.9, 200)
    # sd.lbp(0.85, 20)
    # sd = SimDetector("img/lattice/p4.png", 450, 470, 620, 660, 535, 565)
    # sd.draw_edges(450, 470)
    # sd.ncc_with_cv(0.9935, 150)
