from mxnet import nd
from tools.data_loader import Rotate_Img
import math, cv2
import matplotlib.pyplot as plt


class optimize(object):
    '''
    keypoint, box, img
    to recitfy the face, make it close to vertical
    and detect face landmark again
    Hoever, the total iteration should be low, such as twice
    '''

    def __init__(self, keypoint):
        super(optimize, self).__init__()
        self.keypoint = keypoint[:]
        self.rotate = Rotate_Img()

    def roll_angel(self):
        '''10 points now'''
        dy = self.keypoint[5] - self.keypoint[6]
        dx = self.keypoint[0] - self.keypoint[1]
        # inter_ocu = nd.sqrt(dx ** 2 + dy ** 2)
        # y is vertical inversion
        self.angel = math.atan(dy / dx) * 180 / math.pi

    def transform(self, box, img):
        x1, x2, y1, y2 = box
        self.centerX, self.centerY = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.keypoint[:len(self.keypoint) / 2] = map(lambda x: x * w + x1, self.keypoint[:len(self.keypoint) / 2])
        self.keypoint[len(self.keypoint) / 2:] = map(lambda x: x * h + y1, self.keypoint[len(self.keypoint) / 2:])
        self.roll_angel()
        if abs(self.angel) < 10: return
        new_box = self.trans_point(box, w, h)
        new_box2 = self.trans_point([x1, x2, y2, y1], w, h)
        new_point = self.trans_point(self.keypoint, w, h)
        new_box[len(new_box) / 2:len(new_box) / 2] = new_box2
        rotate_img = self.rotate.ImgRotate(img, self.angel)
        width, height, _ = rotate_img.shape
        x1, x2 = max(0, min(new_box[:len(new_box) / 2])), min(width, max(new_box[:len(new_box) / 2]))
        y1, y2 = max(0, min(new_box[len(new_box) / 2:])), min(height, max(new_box[len(new_box) / 2:]))
        cv2.rectangle(rotate_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.rectangle(img, (int(box[0]), int(box[2])), (int(box[1]), int(box[3])), (255, 0, 0), 2)
        self.show(rotate_img, new_point)
        self.show(img, self.keypoint)
        return rotate_img

    def detransform(self, new_point, new_box):
        pass

    def check_iters(self):
        pass

    def trans_point(self, boxes, w, h):
        new_boxes = [0] * len(boxes)
        for i in range(len(boxes) / 2):
            x, y = boxes[i], boxes[i + len(boxes) / 2]
            newX, newY = self.rotate.getPointAffinedPos(x, y, h, w, self.angel)
            new_boxes[i], new_boxes[i + len(boxes) / 2] = newX, newY
        return new_boxes

    def show(self, img, point):
        for i in range(len(point) / 2):
            x, y = point[i], point[i + len(point) / 2]
            cv2.circle(img, (int(x), int(y)), 1, (255, 255, 255), 2)
        plt.imshow(img)
        plt.show()
