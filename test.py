import cv2
import numpy as np
from pathlib import Path


def project_y(filename: str, show=False, save=True):
    # read, binarization, plot
    img = cv2.imread(filename, 0)
    h, w = img.shape
    _, binary_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("binarization_img", binary_img) if show else None
    cv2.imwrite("img/" + Path(filename).stem + "-binary.jpg", binary_img) if save else None

    # project black points and plot
    new_img = np.full((h, w), fill_value=255, dtype=np.uint8)
    for i in range(0, h):
        black_idx = 0
        for j in range(0, w):
            if binary_img[i, j] == 0:  # point is black
                new_img[i, black_idx] = 0
                black_idx = black_idx + 1
    cv2.imshow("project_to_y_axis", new_img) if show else None
    cv2.imwrite("img/" + Path(filename).stem + "-shadow.jpg", new_img) if save else None

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def project_x(filename: str, show=False, save=True):
    # read, binarization, plot
    img = cv2.imread(filename, 0)
    h, w = img.shape
    _, binary_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("binarization_img", binary_img) if show else None
    cv2.imwrite("img/" + Path(filename).stem + "-binary.jpg", binary_img) if save else None

    # project black points and plot
    new_img = np.full((h, w), fill_value=255, dtype=np.uint8)
    for j in range(0, w):
        black_idx = 0
        for i in range(0, h):
            if binary_img[i, j] == 0:  # point is black
                new_img[h - black_idx - 1, j] = 0
                black_idx = black_idx + 1
    cv2.imshow("project_to_y_axis", new_img) if show else None
    cv2.imwrite("img/" + Path(filename).stem + "-shadow.jpg", new_img) if save else None

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_edges(filename):
    img = cv2.imread(filename)

    # 高斯模糊
    gaussian_img = cv2.GaussianBlur(img, (3, 3), 0)
    # cv2.imshow("gaussian_img", gaussian_img)

    edges = cv2.Canny(gaussian_img, 100, 480)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 0, 0), 1)

    cv2.imshow('edges+lines', img)
    # cv2.imshow('edges+lines', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example2():
    img = cv2.imread('img/test.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 二值化
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 寻找二值图像的轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    project_y("img/2.jpg")
