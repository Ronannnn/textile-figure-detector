from cv2 import ORB, BFMatcher

from img_handle import ImgHandle


class KeyPoint:
    def __init__(self, filename, l_x, l_y, r_x, r_y, c_x, c_y):
        self.pt_h = r_x - l_x
        self.pt_w = r_y - l_y
        self.c_h = c_x - l_x
        self.c_w = c_y - l_y

        self.img_handle = ImgHandle(filename, filter_high_pass=False)
        self.raw_img = self.img_handle.raw_img
        self.gray_img = self.img_handle.img
        self.template = self.gray_img[l_x: r_x, l_y: r_y]

        self.circle_thickness = 1

    def match(self):
        orb = ORB()
        key_points1, desc1 = orb.detectAndCompute(self.raw_img, None)
        key_points2, desc2 = orb.detectAndCompute(self.template, None)
        bf = BFMatcher()
        matches = bf.knnMatch(desc2, desc1, k=100)
        print(matches)
        good = []
        for m, n in matches:
            if m.distance / n.distance < 0.75:
                good.append(m)
        print(good)


if __name__ == '__main__':
    kp = KeyPoint("../img/lattice/p1.png", 45, 48, 90, 90, 67, 69)
    kp.match()
