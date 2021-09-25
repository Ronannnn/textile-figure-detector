import os
import time

import cv2 as cv
import numpy as np
from pathlib import Path
from keras.layers import AveragePooling2D
from skimage.color import rgb2hsv
import skimage.color


class ATM:
    def __init__(
            self,
            filename,
            canny_thresh1=50,
            canny_thresh2=80,
    ):
        filepath = Path(filename)
        self.parent_dir = filepath.parent.absolute()
        self.fn_stem = filepath.stem  # filename without parent dir and suffix
        self.fn_suffix = filepath.suffix  # suffix with dot, e.g. ".jpg"

        self.raw_img = cv.imread(filename)
        # self.hsv_img = rgb2hsv(self.raw_img)
        self.hsv_img = skimage.color.convert_colorspace(self.raw_img, 'RGB', 'HSV')
        self.h_img = self.hsv_img[:, :, 0]
        self.h_img = (self.h_img * 255).astype(np.uint8)  # convert to uint8 before canny
        self.gray_scaled_img = cv.cvtColor(self.raw_img, cv.COLOR_RGB2GRAY)
        # according to the essay
        # n and m are the height and width of the original input image
        # h and w are the height and width of the DTI
        self.n, self.m, _ = self.raw_img.shape
        self.h, self.w = max(20, int(self.n / 20)), max(20, int(self.m / 20))

        # input parameters
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2

        # built-in parameters
        self.edge_thickness = 20
        self.circle_thickness = -1
        self.np_save_delimiter = ", "
        self.np_save_txt_filename = 'edge-density-%s.txt' % self.fn_stem

    def calculate_edge_density(self):
        edges = cv.Canny(self.h_img, self.canny_thresh1, self.canny_thresh2)
        self.__show_and_save(edges, "with-edges")
        # convert 255 to 1.0
        edges_for_pooling = np.float64(edges > 0)
        edges_for_pooling = np.reshape(edges_for_pooling, [1, self.n, self.m, 1])

        # Average Pool Layer
        time_start = time.time()
        print("start at", time_start)
        layer = AveragePooling2D(pool_size=(self.h, self.w), strides=(1, 1), padding='valid')
        edge_density_arr = layer(edges_for_pooling)
        time_end = time.time()
        print("end at", time_end)
        print("total cost", time_end - time_start)

        edge_density_arr = np.reshape(edge_density_arr, [self.n - self.h + 1, self.m - self.w + 1])
        np.savetxt(fname=self.np_save_txt_filename, X=edge_density_arr, fmt='%.10f', delimiter=self.np_save_delimiter)

    def analyze(self, thresh):
        edge_density_arr = np.loadtxt(self.np_save_txt_filename, delimiter=self.np_save_delimiter)
        max_edge_density_arr = edge_density_arr > (thresh * np.max(edge_density_arr))
        print(np.count_nonzero(max_edge_density_arr == 1))
        self.__draw_circles(max_edge_density_arr)

    def __draw_circles(self, points, color=(0, 0, 255)):
        new_img = self.raw_img.copy()
        for i in range(len(points)):
            for j in range(len(points[i])):
                if points[i][j] == 1:
                    # NOTE: coordinate in cv img is different from that in array
                    real_i = i + int(self.h / 2)
                    real_j = j + int(self.w / 2)
                    new_img = cv.circle(new_img, (real_j, real_i), radius=0, color=color, thickness=self.circle_thickness)
        self.__show_and_save(new_img, "with-circles")

    def __show_and_save(self, img, label, show=False, save=True):
        cv.imshow(label, img) if show else None
        cv.imwrite(os.path.join(self.parent_dir, self.fn_stem + "[" + label + "]" + self.fn_suffix), img) if save else None


if __name__ == '__main__':
    atm = ATM("img/lattice/9.png")
    atm.calculate_edge_density()
    atm.analyze(0.9)
