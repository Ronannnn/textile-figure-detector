import cv2 as cv
import numpy as np
from pathlib import Path
import os


class StripeDetector:
    def __init__(
        self,
        filename,
        canny_thresh1=50,
        canny_thresh2=200,
        interval=25,
        merge_thresh=5
    ):
        # filename info
        filepath = Path(filename)
        self.parent_dir = filepath.parent.absolute()
        self.fn_stem = filepath.stem      # filename without parent dir and suffix
        self.fn_suffix = filepath.suffix  # suffix with dot, e.g. ".jpg"

        # img preprocess
        self.raw_img = cv.imread(filename)
        self.shadow_weaken_img = self.__weaken_shadow(self.raw_img)
        self.contrast_enhanced_img = self.__enhance_contrast(self.shadow_weaken_img)
        self.gray_scaled_img = cv.cvtColor(self.contrast_enhanced_img, cv.COLOR_RGB2GRAY)

        # input parameters
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.interval = interval
        self.merge_thresh = merge_thresh

        # built-in parameters
        self.edge_thickness = 1
        self.circle_thickness = -1

    @staticmethod
    def detect_dir(target_dir):
        for _, _, file_list in os.walk(target_dir):
            for filename in file_list:
                if "[" not in filename and "]" not in filename:  # since
                    StripeDetector(
                        filename=os.path.join(target_dir, filename),
                        canny_thresh1=50,
                        canny_thresh2=200,
                        interval=25,
                        merge_thresh=5
                    ).draw_circles_with_canny()

    def draw_circles_with_canny(self):
        edges = self.__get_edges_with_canny(self.canny_thresh1, self.canny_thresh2)
        points = self.__find_coordinates(edges, self.interval, self.merge_thresh)
        self.__draw_circles(points)

    def __get_edges_with_canny(self, thresh1, thresh2, color=(255, 0, 0)):
        edges = cv.Canny(self.gray_scaled_img, thresh1, thresh2)
        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        new_img = self.raw_img.copy()
        cv.drawContours(new_img, contours, -1, color, self.edge_thickness)
        self.__show_and_save(new_img, "with-edges", save=False)
        return edges

    # TODO: suppose stripes are vertical
    def __find_coordinates(self, edges, interval, merge_thresh):
        h, _ = edges.shape
        cur_h = interval
        cloned_edges = np.array(edges)
        cloned_edges[cloned_edges == 255] = 1  # for cv.reduce
        points = []
        while cur_h < h:
            shadow_list = cv.reduce(cloned_edges[cur_h - interval: cur_h, :], 0, cv.REDUCE_SUM, dtype=cv.CV_32S)[0]
            peak = self.__get_peak(shadow_list)
            peak = self.__merge_peak(peak, merge_thresh)
            for x in peak:
                points.append([x, cur_h])
            cur_h = cur_h + interval
        return points

    def __draw_circles(self, points, color=(0, 0, 255)):
        new_img = self.raw_img.copy()
        for point in points:
            new_img = cv.circle(new_img, (point[0], point[1]), radius=0, color=color, thickness=self.circle_thickness)
        self.__show_and_save(new_img, "with-circles")

    @staticmethod
    def __get_peak(shadow_list):
        """
        @param shadow_list: one direction list
        """
        shadow_list = np.array(shadow_list)
        threshold = int(np.max(shadow_list) * 0.25)
        return [i for i in range(len(shadow_list)) if shadow_list[i] > threshold]

    @staticmethod
    def __merge_peak(peak, merge_thresh):
        """
        If the difference of abscissas of two points is smaller than threshold, 
            substitute these two points as their center
        @param peak: one direction array, which store abscissas of points
        """
        if len(peak) == 0:
            return peak
        prev = peak[0]
        ret = []
        for i in range(1, len(peak)):
            cur = peak[i]
            if cur - prev > merge_thresh:
                ret.append(prev)
                prev = cur
            else:
                prev = int((cur + prev) / 2)
        return ret

    @staticmethod
    def __weaken_shadow(img):
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
    def __enhance_contrast(img):
        dst = np.zeros_like(img)
        cv.normalize(img, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        return dst

    def __show_and_save(self, img, label, show=False, save=True):
        cv.imshow(label, img) if show else None
        cv.imwrite(os.path.join(self.parent_dir, self.fn_stem + "[" + label + "]" + self.fn_suffix), img) if save else None


if __name__ == '__main__':
    StripeDetector.detect_dir("img/stripe/raw")
    StripeDetector("img/stripe/raw/a6.png", merge_thresh=1).draw_circles_with_canny()
