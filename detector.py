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
        h, _ = binary_img.shape
        for j in range(len(black_dot_sum)):
            for i in range(h - black_dot_sum[j], h):
                new_img[i, j] = 0
        self.__show_and_save(new_img, "shadow-x")
        self.__cv2_join()

    def draw(self, contours, color=(0, 0, 0)):
        cloned_raw_img = self.raw_img.copy()
        cv2.drawContours(cloned_raw_img, contours, -1, color, 5)
        return cloned_raw_img

    def test_draw(self):
        """draw the central line"""
        h, w, _ = self.raw_img.shape
        print(h, w)
        test = []
        for i in range(h):
            test.append([[int(w / 2), i]])
        print([np.array(test)])
        draw_img = self.draw([np.array(test)])
        self.__show_and_save(draw_img, "draw-img")

    def __get_binary_img(self):
        _, binary_img = cv2.threshold(self.gray_scaled_img, 150, 255, cv2.THRESH_BINARY)
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
    TextileDetector("4.jpg").test_draw()
