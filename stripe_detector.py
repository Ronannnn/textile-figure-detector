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
        self.shadow_weaken_img = self.__weaken_shadow(self.raw_img)
        self.contrast_enhanced_img = self.__enhance_contrast(self.shadow_weaken_img)
        self.gray_scaled_img = cv.cvtColor(self.contrast_enhanced_img, cv.COLOR_RGB2GRAY)
        self.background = self.shadow_weaken_img

        self.thickness = 1

    def draw_edges_with_canny(self, thresh1, thresh2, color=(255, 0, 0)):
        edges = cv.Canny(self.gray_scaled_img, thresh1, thresh2)
        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours = self.__remove_invalid_edges(contours, 500)
        new_img = self.background.copy()
        cv.drawContours(new_img, contours, -1, color, 2)
        self.__show_and_save(new_img, "edges_with_canny")
        return edges

    # TODO: suppose stripes are vertical
    def find_coordinate(self, edges, interval):
        h, _ = edges.shape
        cur_h = interval
        cloned_edges = np.array(edges)
        cloned_edges[cloned_edges == 255] = 1  # for cv.reduce
        points = []
        while cur_h < h:
            shadow_list = cv.reduce(cloned_edges[cur_h - interval: cur_h, :], 0, cv.REDUCE_SUM, dtype=cv.CV_32S)[0]
            peak = self.__get_peak_2(shadow_list)
            # peak = [peak[i] for i in range(len(peak)) if i % 2 == 0]
            for x in peak:
                points.append([x, cur_h])
            cur_h = cur_h + interval
        return points

    def draw_circles(self, points):
        new_img = self.raw_img.copy()
        for point in points:
            new_img = cv.circle(new_img, (point[0], point[1]), radius=0, color=(0, 0, 255), thickness=-1)
        self.__show_and_save(new_img, "canny_with_circles")

    @staticmethod
    def __get_peak_1(shadow_list):
        shadow_list = np.array(shadow_list)
        threshold = int(np.max(shadow_list) * 0.25)
        return [i for i in range(len(shadow_list)) if shadow_list[i] > threshold]
    
    @staticmethod
    def __get_peak_2(shadow_list):
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

    def draw_edges_with_structure(self, thresh, color=(255, 0, 0)):
        _, img = cv.threshold(self.gray_scaled_img, thresh, 255, cv.THRESH_BINARY)  # 设定红色通道阈值210（阈值影响梯度运算效果）
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))  # 定义矩形结构元素
        dilated = cv.dilate(img, kernel)
        contours, _ = cv.findContours(dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        new_img = self.background.copy()
        cv.drawContours(new_img, contours, -1, color, self.thickness)
        self.__show_and_save(new_img, "edges_with_structure")
        return contours

    @staticmethod
    def __remove_invalid_edges(contours, invalid_thresh):
        ret = []
        for line in contours:
            if (len(line) > invalid_thresh):
                ret.append(line)
        return ret

    @staticmethod
    def replenish_contours(contours):
        print(len(contours))
        # contours = np.vstack(contours).squeeze()
        print(contours)

    # Ref: https://www.codenong.com/44752240/
    def __weaken_shadow(self, img):
        rgb_planes = cv.split(img)
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv.medianBlur(dilated_img, 21)
            diff_img = 255 - cv.absdiff(plane, bg_img)
            norm_img = cv.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
            result_norm_planes.append(norm_img)
        result_norm = cv.merge(result_norm_planes)
        self.__show_and_save(result_norm, "weaken-shadow")
        return result_norm

    def __enhance_contrast(self, img):
        dst = np.zeros_like(img)
        cv.normalize(img, dst, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        self.__show_and_save(dst, "enhanced-contrast")
        return dst

    def __show_and_save(self, img, label, show=False, save=True):
        cv.imshow(label, img) if show else None
        cv.imwrite(os.path.join(self.parent_dir, self.fn_stem + "-" + label + self.fn_suffix), img) if save else None

    @staticmethod
    def __cv_join():
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    sd = StripeDetector("img/stripe/2.png")
    edges = sd.draw_edges_with_canny(90, 100)
    points = sd.find_coordinate(edges, 100)
    sd.draw_circles(points)
    # incomplete_contours = sd.draw_edges_with_structure(20)
    # complete_contours = sd.replenish_contours(incomplete_contours)
    # sd.remove_shadow_by_max_filtering()
