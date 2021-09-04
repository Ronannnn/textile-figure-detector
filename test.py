import cv2
import numpy as np
from pathlib import Path


def project_y(filename: str, show=False, save=True):
    # read, binarization, plot
    img = cv2.imread(filename, 0)
    h, w = img.shape
    _, binary_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    binary_img = adjust_black(binary_img)
    cv2.imshow("binarization_img", binary_img) if show else None
    cv2.imwrite("img/" + Path(filename).stem + "-binary.jpg", binary_img) if save else None

    # project black points and plot
    new_img = np.full((h, w), fill_value=255, dtype=np.uint8)  # create a new img
    black_dot_sum = np.sum(binary_img == 0, axis=1)  # sum all black dots
    for i in range(len(black_dot_sum)):
        for j in range(0, black_dot_sum[i]):
            new_img[i, j] = 0
    cv2.imshow("project_to_y_axis", new_img) if show else None
    cv2.imwrite("img/" + Path(filename).stem + "-shadow.jpg", new_img) if save else None

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def project_x(filename: str, show=False, save=True):
    # read, binarization, plot
    img = cv2.imread(filename, 0)
    h, w = img.shape
    _, binary_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    binary_img = adjust_black(binary_img)
    cv2.imshow("binarization_img", binary_img) if show else None
    cv2.imwrite("img/" + Path(filename).stem + "-binary.jpg", binary_img) if save else None

    # project black points and plot
    new_img = np.full((h, w), fill_value=255, dtype=np.uint8)
    black_dot_sum = np.sum(binary_img == 0, axis=0)
    for j in range(len(black_dot_sum)):
        for i in range(h - black_dot_sum[j], h):
            new_img[i, j] = 0
    cv2.imshow("project_to_y_axis", new_img) if show else None
    cv2.imwrite("img/" + Path(filename).stem + "-shadow.jpg", new_img) if save else None

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def adjust_black(binary_img):
    black_dot = np.sum(binary_img)
    h, w = binary_img.shape
    if black_dot > h * w / 2:
        binary_img = np.where(binary_img == 0, 255, 0)
    return binary_img.astype(np.uint8)


def draw_edges_canny(filename):
    img = cv2.imread(filename)
    gaussian_img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯模糊

    edges = cv2.Canny(gaussian_img, 100, 480)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 0, 0), 1)
    cv2.imshow('edges+lines', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_edges_binarization():
    img = cv2.imread('img/test.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 二值化
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 寻找二值图像的轮廓

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    cv2.imshow('result', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    project_x("img/4.jpg")
