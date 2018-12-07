import sys, dlib, cv2, torch, os, random
from torch.autograd import Variable
import torch.nn.functional as F
from dapeng.dp_detect import Detector

sys.path.append('/home/ldc/work/faceboxes/')
from networks import FaceBox
from encoderl import DataEncoder


# import torch#only work in python3 &corresponding pytorch
# torch.multiprocessing.set_start_method("spawn")

class Face_Roi(object):
    def __init__(self, fun='facebox'):
        '''facebox / dlib'''
        self.mode = fun
        self.detector = ''
        self.net = ''
        self.dp = ''

    def detect(self, img, point, image_size):
        # res=self.dlib_detect(img)
        try:
            if self.mode == 'facebox':
                res = self.facebox_detect(img)
            elif self.mode == 'dlib':
                res = self.dlib_detect(img)
            else:
                res = self.dapeng_detect(img)
            idx = self.IOU(res, point)
            box = res[idx]
        except Exception, e:
            box = self.boxes_alter(img, point)
            box = self.check(box, point, img, image_size)
        # box = self.zooming(box)
        box = self.check(box, point, img, image_size)
        return box

    def get_dlib_detector(self):
        path_to_detector = "/home/ldc/work/faceAlignment/face-alignment/test/mmod_human_face_detector.dat"
        self.detector = dlib.cnn_face_detection_model_v1(path_to_detector)

    def dlib_detect(self, img):
        if self.detector == '':
            self.get_dlib_detector()
        # print 'enter dect 14',img.shape,img.dtype
        dets = self.detector(img, 1)
        # print 'step over 15'
        res = []
        for index, det in enumerate(dets):
            det = det.rect
            x1 = det.left()
            y1 = det.top()
            x2 = det.right()
            y2 = det.bottom()
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > img.shape[1]: x2 = img.shape[1]
            if y2 > img.shape[0]: y2 = img.shape[0]
            res.append([x1, x2, y1, y2])
        return res

    def IOU(self, rec, point):
        if len(point) > 4:
            L = len(point) / 2
            x1, x2, y1, y2 = min(point[0:L]), max(point[0:L]), min(point[L:]), max(point[L:])
        else:
            x1, x2, y1, y2 = point
            # print 'point range:',x1,x2,y1,y2
        res = []
        for i, r in enumerate(rec):
            x11, x22, y11, y22 = r
            rx1, ry1 = max(x1, x11), max(y1, y11)
            rx2, ry2 = min(x2, x22), min(y2, y22)
            rw, rh = rx2 - rx1, ry2 - ry1
            if rw <= 0 or rh <= 0:
                res.append(0.0)
                continue
            w1, h1 = x2 - x1, y2 - y1
            w11, h22 = x22 - x11, y22 - y11
            res.append(1.0 * (rw * rh) / (w1 * h1 + w11 * h22 - rw * rh))
        # print 'IOU:',res
        return res.index(max(res))

    # -  -  -  -  -  face box  -  -  -  -  -  -  -  -  -  -  -
    def facebox_detect(self, img):
        if self.net == '':
            self.get_facebox_detector()
        im = cv2.resize(img, (1024, 1024))
        im_tensor = torch.from_numpy(im.transpose((2, 0, 1)))
        im_tensor = im_tensor.float().div(255)
        # print(im_tensor.shape)
        # loc, conf = self.net(Variable(torch.unsqueeze(im_tensor, 0), volatile=True).cuda())
        with torch.no_grad():
            loc, conf = self.net(Variable(torch.unsqueeze(im_tensor, 0), ).cuda(1))
        loc, conf = loc.cpu(), conf.cpu()
        boxes, labels, probs = self.data_encoder.decode(loc.data.squeeze(0),
                                                        F.softmax(conf.squeeze(0)).data)
        h, w, _ = img.shape
        for box in boxes:
            x1 = box[0] * w  # later: x1
            x2 = box[2] * w  # later: x2
            y1 = box[1] * h  # later: y1
            y2 = box[3] * h  # later: y2
            box[0], box[1], box[2], box[3] = x1, x2, y1, y2
        return boxes

    def get_facebox_detector(self):
        pt = "/home/ldc/work/faceboxes/ckpt/faceboxes.pt"
        self.net = FaceBox()
        self.net.load_state_dict(torch.load(pt, map_location=lambda storage, loc: storage))
        # print torch.cuda.is_available()
        # torch.manual_seed(23)
        self.net.cuda(1)
        self.net.eval()
        self.data_encoder = DataEncoder()

    def get_dapeng_detector(self):
        dll = "/home/ldc/work/faceAlignment/TCDCN/dapeng/libface_detect_x64.so"
        self.dp = Detector(dll_path=dll)

    def dapeng_detect(self, img):
        if self.dp == '':
            self.get_dapeng_detector()
        rets = self.dp.detect(img)
        return rets

    def boxes_alter(self, img, point):
        print '*' * 10, 'alternative box: ',
        # return a box contain a face
        h, w, _ = img.shape
        expand_x = 1.8
        expand_y = 1.7
        # left - right
        inter_ocu = abs(point[1] - point[0])
        ratio_ocu = abs(1.0 * (point[2] - point[0]) / (point[0] - point[1]))
        left = point[0] - inter_ocu * ratio_ocu * expand_x  # inter_ocu*ratio_ocu: left differ to right
        if ratio_ocu < 1.0:
            right = point[1] + inter_ocu * (1.0 - ratio_ocu) * expand_x
        else:
            right = point[2] + inter_ocu * (ratio_ocu - 1.0) * expand_x
            # top - bottom
        eyebrows = (point[5] + point[6]) / 2.0
        mouse = (point[8] + point[9]) / 2.0
        inter_face = abs(eyebrows - mouse)
        ratio_face = abs(1.0 * (eyebrows - point[7]) / inter_face)
        top = eyebrows - inter_face * ratio_face * expand_y
        bottom = mouse + inter_face * (1.0 - ratio_face) * expand_y
        return left, right, top, bottom

    def check(self, box, point, img, image_size):
        L = len(point) / 2
        x1, x2, y1, y2 = box
        # box & img
        h, w, _ = img.shape
        # box & point
        left, right = min(point[0:L]), max(point[0:L])
        top, bottom = min(point[L:]), max(point[L:])
        # off at least 4 pixels for that kernel size equals 3 in third convolution
        off_last_col, off_last_line = 4 * image_size / (right - left), 4 * image_size / (bottom - top)
        right += off_last_col
        bottom += off_last_line
        x1 = max(min(x1, left), 0)
        x2 = min(max(x2, right), w - 1)
        y1 = max(min(y1, top), 0)
        y2 = min(max(y2, bottom), h - 1)
        return x1, x2, y1, y2

    def zooming(sefl, box):
        x1, x2, y1, y2 = box
        w_box, h_box = x2 - x1, y2 - y1
        ratio = random.random() * 0.05  # range of widht or hegiht
        x1, x2 = x1 - w_box * ratio, x2 + w_box * ratio
        y1, y2 = y1 - h_box * ratio, y2 + h_box * ratio
        return x1, x2, y1, y2


def file_list_fn(path):
    file_list = []
    files = os.listdir(path)
    for f in files:
        if 'jpg' not in f: continue
        file_list.append(f)
    return file_list


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    images_dir = "/home/ldc/work/faceAlignment/face-alignment/test/assets/"
    image_list = file_list_fn(images_dir)
    detector = Face_Roi()
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

    for image in image_list:
        im = cv2.imread(images_dir + image)
        boxes = detector.facebox_detect(im)
        for i, (box) in enumerate(boxes):
            print('i', i, 'box', box)
            x1, x2, y1, y2 = box
            # x1 = box[0]
            # x2 = box[2]
            # y1 = box[1]
            # y2 = box[3]
            cv2.rectangle(im, (x1, y1 + 4), (x2, y2), (0, 0, 255), 2)
            cv2.putText(im, 'face', (x1, y1), font, 0.4, (0, 0, 255))
        out = 'result/' + image
        print out
        cv2.imwrite(out, im)
