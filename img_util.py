import os
from pathlib import Path

import numpy as np
import cv2 as cv


class ImgUtil:
    @staticmethod
    def filter_high_pass(img, filter_h):
        """
        NOTE: filter_w will be generated automatically according to the shape of the original figure
        """
        # fourier transform
        f = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f)

        # set high pass filter
        h, w = img.shape
        center_x, center_y = int(h / 2), int(w / 2)
        filter_w = int(filter_h * w / h)
        h_half = int(filter_h / 2)
        w_half = int(filter_w / 2)
        f_shift[center_x - h_half:center_x + h_half, center_y - w_half:center_y + w_half] = 0

        # inverse fourier transform
        i_shift = np.fft.ifftshift(f_shift)
        i_img = np.fft.ifft2(i_shift)
        i_img = np.abs(i_img)

        return i_img

    @staticmethod
    def enhance_contrast(img):
        dst = np.zeros_like(img)
        cv.normalize(img, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        return dst

    @staticmethod
    def draw_rectangle(img, x, y, h, w, color=(0, 0, 255), thickness=1):
        pts = []
        # top and bottom
        for i in range(y, y + w + 1):
            pts.append((x, i))
            pts.append((x + h, i))
        # left and right
        for i in range(x + 1, x + h):
            pts.append((i, y))
            pts.append((i, y + w))
        for p in pts:
            img = cv.circle(img, (p[1], p[0]), radius=0, color=color, thickness=thickness)
        return img

    @staticmethod
    def save(img, filename, label):
        filepath = Path(filename)
        labeled_filename = os.path.join(filepath.parent.absolute(), "%s[%s]%s" % (filepath.stem, label, filepath.suffix))
        cv.imwrite(labeled_filename, img)

    @staticmethod
    def show(img, label):
        cv.imshow(label, img)
