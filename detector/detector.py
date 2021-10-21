import cv2 as cv
import numpy as np
from pathlib import Path
import os

from img_handle import ImgHandle


class TextileDetector:
    def __init__(self, filename):
        self.img_handle = ImgHandle(filename, weaken_shadow=False, enhance_contrast=False, to_gray=False)

        self.raw_img = self.img_handle.raw_img
        self.gray_img = self.img_handle.img

    def project_y(self):
        binary_img = cv.threshold(self.gray_img, 140, 255, cv.THRESH_BINARY)
        self.img_handle.save(binary_img, "binary-y")
        new_img = np.full(binary_img.shape, fill_value=255, dtype=np.uint8)  # create a new img with white dots
        black_dot_sum = np.sum(binary_img == 0, axis=1)  # sum all black dots
        peak_y = self.__get_peak(black_dot_sum)
        draw_img = self.__draw_with_one_d_list(peak_y, axis='y')
        self.img_handle.save(draw_img, "draw-img-y")
        for i in range(len(black_dot_sum)):
            for j in range(0, black_dot_sum[i]):
                new_img[i, j] = 0  # plot black dot
        self.img_handle.save(new_img, "shadow-y")
        self.img_handle.cv_join()

    def project_x(self):
        binary_img = cv.threshold(self.gray_img, 140, 255, cv.THRESH_BINARY)
        self.img_handle.save(binary_img, "binary-x")
        new_img = np.full(binary_img.shape, fill_value=255, dtype=np.uint8)  # create a new img
        black_dot_sum = np.sum(binary_img == 0, axis=0)
        peak_x = self.__get_peak(black_dot_sum)
        draw_img = self.__draw_with_one_d_list(peak_x, axis='x')
        self.img_handle.save(draw_img, "draw-img-x")
        h, _ = binary_img.shape
        for j in range(len(black_dot_sum)):
            for i in range(h - black_dot_sum[j], h):
                new_img[i, j] = 0
        self.img_handle.save(new_img, "shadow-x")
        self.img_handle.cv_join()

    def filter_high_pass(self, filter_h):
        """
        NOTE: filter_w will be generated automatically according to the shape of the original figure
        """
        # fourier transform
        f = np.fft.fft2(self.gray_img)
        f_shift = np.fft.fftshift(f)

        # set high pass filter
        h, w = self.gray_img.shape
        center_x, center_y = int(h / 2), int(w / 2)
        filter_w = int(filter_h * w / h)
        h_half = int(filter_h / 2)
        w_half = int(filter_w / 2)
        f_shift[center_x - h_half:center_x + h_half, center_y - w_half:center_y + w_half] = 0

        # inverse fourier transform
        i_shift = np.fft.ifftshift(f_shift)
        i_img = np.fft.ifft2(i_shift)
        i_img = np.abs(i_img)

        self.img_handle.save(i_img, "filter_high_pass")
        return i_img

    def draw_edges_with_canny(self, thresh1, thresh2, color=(255, 0, 0)):
        edges = cv.Canny(self.gray_img, thresh1, thresh2)
        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        new_img = self.raw_img.copy()
        cv.drawContours(new_img, contours, -1, color, 3)
        self.img_handle.save(new_img, "edges_with_canny")
        return contours

    def draw_edges_with_structure(self, thresh):
        _, img = cv.threshold(self.gray_img, thresh, 255, cv.THRESH_BINARY)  # 设定红色通道阈值210（阈值影响梯度运算效果）
        self.img_handle.save(img, "edges_with_structure_test")
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 定义矩形结构元素
        gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)  # 梯度
        self.img_handle.save(gradient, "edges_with_structure")

    @staticmethod
    def __get_peak(shadow_list):
        shadow_list = np.array(shadow_list)
        threshold = int(np.max(shadow_list) * 0.75)
        shrink_list = [i if i > threshold else 0 for i in shadow_list]
        peak = []
        i = 0
        while i < len(shrink_list):
            if shrink_list[i] == 0:
                i = i + 1
                continue
            start = i
            end = i
            break_count = 0
            while break_count < 50 and i < len(shrink_list):
                if shrink_list[i] == 0:
                    break_count = break_count + 1
                else:
                    end = i
                    break_count = 0
                i = i + 1
            if end - start < 5:
                continue
            peak.append(int((start + end) / 2))
        print(len(peak))
        return peak

    def __draw_with_one_d_list(self, one_direction_list, axis, color=(0, 0, 0)):
        cloned_raw_img = self.raw_img.copy()
        contours = self.__get_contours(one_direction_list, axis=axis, shape=cloned_raw_img.shape)
        cv.drawContours(cloned_raw_img, contours, -1, color, 2)
        return cloned_raw_img

    @staticmethod
    def __get_contours(one_direction_list, axis, shape):
        h, w, _ = shape
        contours = []
        if axis == 'x':
            for i in range(len(one_direction_list)):
                line = []
                for h_idx in range(h):
                    line.append([[int(one_direction_list[i]), h_idx]])
                contours.append(np.array(line))
        elif axis == 'y':
            for i in range(len(one_direction_list)):
                line = []
                for w_idx in range(w):
                    line.append([[w_idx, int(one_direction_list[i])]])
                contours.append(np.array(line))
        else:
            raise Exception("axis not supported")
        return contours


if __name__ == '__main__':
    # TextileDetector("img/5.jpg").filter_high_pass(400)
    # TextileDetector("img/6.jpg").draw_edges_with_structure(80)
    TextileDetector("../img/stripe/5.png").draw_edges_with_canny(20, 200)
