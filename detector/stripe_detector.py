import cv2 as cv
import numpy as np
import os

from img_handle import ImgHandle


class StripeDetector:
    def __init__(
            self,
            filename,
            is_vertical,
            canny_thresh1=50,
            canny_thresh2=200,
            interval=25,
            merge_thresh=5
    ):
        """
        @param filename: filename can contain parent directory
        @param canny_thresh1: see cv.canny for more details
        @param canny_thresh2: see cv.canny for more details
        @param interval: see __find_coordinates fore more details
        @param merge_thresh: see __merge_peak for more details
        """
        self.img_handle = ImgHandle(filename, filter_high_pass=False)

        # img preprocess
        self.raw_img = self.img_handle.raw_img
        self.gray_img = self.img_handle.img

        # input parameters
        self.is_vertical = is_vertical
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.interval = interval
        self.merge_thresh = merge_thresh

        # built-in parameters
        self.edge_thickness = 1
        self.circle_thickness = -1

    @staticmethod
    def detect_dir(target_dir):
        # iterate all files except files with "[" or "]"
        for _, _, file_list in os.walk(target_dir):
            for filename in file_list:
                if "[" not in filename and "]" not in filename:  # since
                    StripeDetector(
                        filename=os.path.join(target_dir, filename),
                        is_vertical=True,
                        canny_thresh1=50,
                        canny_thresh2=200,
                        interval=25,
                        merge_thresh=5
                    ).draw_circles_with_canny()

    def draw_circles_with_canny(self):
        edges = self.__get_edges_with_canny(self.canny_thresh1, self.canny_thresh2)
        points = self.__find_coordinates(edges, self.is_vertical, self.interval, self.merge_thresh)
        self.__draw_circles(points)

    def __get_edges_with_canny(self, thresh1, thresh2, color=(255, 0, 0)):
        edges = cv.Canny(self.gray_img, thresh1, thresh2)
        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        new_img = self.raw_img.copy()
        cv.drawContours(new_img, contours, -1, color, self.edge_thickness)
        self.img_handle.save(new_img, "with-edges")
        return edges

    # TODO: suppose stripes are vertical
    def __find_coordinates(self, edges, is_vertical, interval, merge_thresh):
        """
        For each interval, project black points and find the local peak for each edge

        @param interval: if stripes are vertical, the interval means the distance of horizontal lines
        @param merge_thresh: see __merge_peak for more details 
        """
        if is_vertical is not True:
            edges = np.transpose(edges)
        h, _ = edges.shape
        cur_h = interval
        cloned_edges = np.array(edges)
        cloned_edges[cloned_edges == 255] = 1  # for cv.reduce
        points = np.zeros(edges.shape)
        while cur_h < h:
            projection_list = cv.reduce(cloned_edges[cur_h - interval: cur_h, :], 0, cv.REDUCE_SUM, dtype=cv.CV_32S)[0]
            peak = self.__get_peak(projection_list)
            peak = self.__merge_peak(peak, merge_thresh)
            for x in peak:
                points[cur_h][x] = 1
            cur_h = cur_h + interval
        if is_vertical is not True:
            points = np.transpose(points)
        return points

    def __draw_circles(self, points, color=(0, 0, 255)):
        new_img = self.raw_img.copy()
        for i in range(len(points)):
            for j in range(len(points[i])):
                if points[i][j] == 1:
                    # NOTE: coordinate in cv img is different from that in array
                    new_img = cv.circle(new_img, (j, i), radius=0, color=color, thickness=self.circle_thickness)
        self.img_handle.save(new_img, "with-circles")

    @staticmethod
    def __get_peak(projection_list):
        """
        @param projection_list: one direction list
        """
        projection_list = np.array(projection_list)
        threshold = int(np.max(projection_list) * 0.25)
        return [i for i in range(len(projection_list)) if projection_list[i] > threshold]

    @staticmethod
    def __merge_peak(peak, merge_thresh):
        """
        If the difference of abscissas of two points is smaller than threshold, 
            substitute these two points with their center
        @param peak: one direction array, which store abscissas of peaks
        @param merge_thresh: each edge may have multiple projection peaks, this func will merge peaks within a certain range
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


if __name__ == '__main__':
    # StripeDetector.detect_dir("img/stripe/raw")
    StripeDetector("../img/stripe/3.png", is_vertical=False, merge_thresh=5).draw_circles_with_canny()
