import os
from pathlib import Path

import numpy as np
import cv2 as cv


class ImgHandle:
    def __init__(self, filename, weaken_shadow=True, enhance_contrast=True, to_gray=True, filter_high_pass=True):
        filepath = Path(filename)
        self._parent_dir = filepath.parent.absolute()
        self._fn_stem = filepath.stem  # filename without parent dir and suffix
        self._fn_suffix = filepath.suffix  # suffix with dot, e.g. ".jpg"

        self._raw_img = cv.imread(filename)
        self._img = self.weaken_shadow(self._raw_img) if weaken_shadow else self._raw_img
        self._img = self.enhance_contrast(self._img) if enhance_contrast else self._img
        self._img = cv.cvtColor(self._img, cv.COLOR_RGB2GRAY) if to_gray else self._img
        self._img = ImgHandle.filter_high_pass(self._img, 2) if filter_high_pass else self._img

    @property
    def raw_img(self):
        return self._raw_img

    @property
    def img(self):
        return self._img

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
    def weaken_shadow(img):
        """
        Ref: https://www.codenong.com/44752240/
        """
        rgb_planes = cv.split(img)
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv.medianBlur(dilated_img, 21)
            diff_img = 255 - cv.absdiff(plane, bg_img)
            norm_img = cv.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
            result_norm_planes.append(norm_img)
        result_norm = cv.merge(result_norm_planes)
        return result_norm

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

    def save(self, img, label):
        cv.imwrite(os.path.join(self._parent_dir, "%s[%s]%s" % (self._fn_stem, label, self._fn_suffix), img))

    @staticmethod
    def show(img, label):
        cv.imshow(label, img)

    @staticmethod
    def cv_join():
        cv.waitKey(0)
        cv.destroyAllWindows()
