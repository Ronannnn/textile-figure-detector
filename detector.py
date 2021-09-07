import cv2
import numpy as np
from pathlib import Path
import os


class TextileDetector:
    def __init__(self, filename, dirname="img"):
        filepath = Path(filename)
        self.fn_stem = filepath.stem
        self.fn_suffix = filepath.suffix  # suffix with dot, e.g. ".jpg"
        self.dirname = dirname

        self.raw_img = cv2.imread(os.path.join(dirname, filename))
        self.gray_scaled_img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2GRAY)

    def project_y(self):
        binary_img = self.__get_binary_img()
        self.__show_and_save(binary_img, "binary-y")
        new_img = np.full(binary_img.shape, fill_value=255, dtype=np.uint8)  # create a new img with white dots
        black_dot_sum = np.sum(binary_img == 0, axis=1)  # sum all black dots
        peak_y = self.__get_peak_2(black_dot_sum)
        print(peak_y)
        draw_img = self.__get_draw_img(peak_y, axis='y')
        self.__show_and_save(draw_img, "draw-img-y")
        for i in range(len(black_dot_sum)):
            for j in range(0, black_dot_sum[i]):
                new_img[i, j] = 0  # plot black dot
        self.__show_and_save(new_img, "shadow-y")
        self.__cv2_join()

    def project_x(self):
        binary_img = self.__get_binary_img()
        self.__show_and_save(binary_img, "binary-x")
        new_img = np.full(binary_img.shape, fill_value=255, dtype=np.uint8)  # create a new img
        black_dot_sum = np.sum(binary_img == 0, axis=0)
        peak_x = self.__get_peak_2(black_dot_sum)
        print(peak_x)
        draw_img = self.__get_draw_img(peak_x, axis='x')
        self.__show_and_save(draw_img, "draw-img-x")
        h, _ = binary_img.shape
        for j in range(len(black_dot_sum)):
            for i in range(h - black_dot_sum[j], h):
                new_img[i, j] = 0
        self.__show_and_save(new_img, "shadow-x")
        self.__cv2_join()

    @staticmethod
    def __get_peak(shadow_list):
        """
        Reference: https://www.cnblogs.com/ronny/p/3616470.html
        """
        # first order difference vector
        vec = shadow_list[1:] - shadow_list[:-1]
        vec = np.sign(vec)
        if vec[len(vec) - 1] == 0:
            vec[len(vec) - 1] = 1
        for i in range(len(vec) - 2, 0, -1):
            if vec[i] == 0:
                if vec[i + 1] >= 0:
                    vec[i] = 1
                else:
                    vec[i] = -1
        vec = vec[1:] - vec[:-1]
        peak = []
        for i in range(len(vec) - 1):
            if vec[i + 1] - vec[i] == -2:  # peak
                peak.append(i + 1)
        return peak

    @staticmethod
    def __get_peak_2(shadow_list):
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

    def __get_draw_img(self, one_direction_list, axis, color=(0, 0, 0)):
        cloned_raw_img = self.raw_img.copy()
        contours = self.__get_contours(one_direction_list, axis=axis, shape=cloned_raw_img.shape)
        cv2.drawContours(cloned_raw_img, contours, -1, color, 2)
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

    def __get_binary_img(self):
        _, binary_img = cv2.threshold(self.gray_scaled_img, 140, 255, cv2.THRESH_BINARY)
        binary_img = self.__adjust_black(binary_img)
        return binary_img

    def __show_and_save(self, img, label, show=False, save=True):
        cv2.imshow(label, img) if show else None
        cv2.imwrite(os.path.join(self.dirname, self.fn_stem + "-" + label + self.fn_suffix), img) if save else None

    @staticmethod
    def __adjust_black(binary_img):
        """
        If the number of black dots is greater than that of white dots, invert black and white
        :param binary_img:
        :return: inverted binary img
        """
        black_dot = np.sum(binary_img)
        h, w = binary_img.shape
        if black_dot > h * w / 2:
            binary_img = np.where(binary_img == 0, 255, 0)
        return binary_img.astype(np.uint8)

    @staticmethod
    def __cv2_join():
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    TextileDetector("4.jpg").project_y()
