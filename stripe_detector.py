import cv2 as cv
import numpy as np
from pathlib import Path
import os


class StripeDetector:
    def __init__(self, filename):
        filepath = Path(filename)
        self.parent_dir = filepath.parent.absolute()
        self.fn_stem = filepath.stem      # filename without parent dir and suffix
        self.fn_suffix = filepath.suffix  # suffix with dot, e.g. ".jpg"

        self.raw_img = cv.imread(filename)
        self.gray_scaled_img = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    def draw_edges_with_canny(self, thresh1, thresh2, color=(255, 0, 0)):
        edges = cv.Canny(self.gray_scaled_img, thresh1, thresh2)
        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        new_img = self.raw_img.copy()
        print(new_img.shape)
        cv.drawContours(new_img, contours, -1, color, 2)
        self.__show_and_save(new_img, "edges_with_canny")
        return contours
    
    @staticmethod
    def replenish_contours(contours):
        print(len(contours))
        # contours = np.vstack(contours).squeeze()
        print(contours)

    def __show_and_save(self, img, label, show=False, save=True):
        cv.imshow(label, img) if show else None
        cv.imwrite(os.path.join(self.parent_dir, self.fn_stem + "-" + label + self.fn_suffix), img) if save else None

    @staticmethod
    def __cv_join():
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    sd = StripeDetector("img/stripe/5.png")
    incomplete_contours = sd.draw_edges_with_canny(20, 200)
    complete_contours = sd.replenish_contours(incomplete_contours)
