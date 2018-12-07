import cv2, os, time
from tools.net import TCDCN
import mxnet as mx
from tools.detector import Face_Roi
from mxnet import nd
from tools.data_loader import database, HDF5_dataset
import numpy as np


# import numpy as np
# from data_loader import watch_image
def file_list_fn(path):
    base = ['png', 'bmp', 'jpg', 'PNG', 'BMP', 'JPG']
    file_list = []
    files = os.listdir(path)
    for f in files:
        if not sum(map(lambda x: x in f, base)):
            continue
        file_list.append(path + f)
    return file_list


def whitering(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, s = cv2.meanStdDev(img)
    img = (img - m) / (1.e-6 + s)
    img = img.astype('float32')
    return img


def Evaluation(y_pred, y_true):
    num, L = y_true.shape
    L /= 2
    delX = y_true[:, 0] - y_true[:, 1]  # del X size 16
    delY = y_true[:, L] - y_true[:, L + 1]  # del y size 16
    interOc = (1e-6 + (delX * delX + delY * delY) ** 0.5).T
    # Cannot multiply shape (16,10) by (16,1) so we transpose to (10,16) and (1,16)
    diff = (y_pred - y_true).T  # Transpose so we can divide a (16,10) array by (16,1)
    diff = (diff / interOc).T  # We transpose back to (16,10)
    res = np.mean(np.fabs(diff.asnumpy()), 1)
    return res


def evalue(file, test_data, Is_hdf5=True):
    '''mean error and failure rate.'''
    net = TCDCN()
    ctx = mx.gpu()
    net.load_parameters(file, ctx=ctx)

    res = []
    for batch in test_data:
        if Is_hdf5: batch = batch.data
        t = time.time()
        data = batch[0].as_in_context(ctx)
        landmark = batch[1].as_in_context(ctx)
        # attr = batch[2].as_in_context(ctx)
        out = net(data)
        tmp = Evaluation(out[0], landmark)
        res += tmp.tolist()
    print
    res = np.array(res)
    print 'mean error: ', np.mean(res)
    print 'failure rate: ', 1 - np.mean(res < 0.1)
    print


class test_model(object):
    def __init__(self, model_path):
        super(test_model, self).__init__()
        self.net = TCDCN()
        self.ctx = mx.gpu()
        self.net.load_parameters(model_path, ctx=self.ctx)
        self.detector = Face_Roi()
        self.IMAGE_SIZE = 40

    def test(self, image_list, use_detector=True):
        for idx, jpg in enumerate(image_list):
            saved_ = jpg.split('.')
            saved_jpg = '_'.join(saved_[:-1] + ['TCNN.jpg'])
            saved_txt = '_'.join(saved_[:-1] + ['TCNN.txt'])
            self.exist_box = '.'.join(saved_[:-1] + ['txt'])
            print saved_jpg

            img = cv2.imread(jpg)
            key_points = self.landmark_of_given_img(img, use_detector=use_detector)
            cv2.imwrite(saved_jpg, img)
            with open(saved_txt, 'w')as f:
                f.writelines(' '.join(map(repr, key_points)))

    def landmark_of_given_img(self, img, use_detector=False, stop=False):
        try:
            if use_detector:
                print 'facebox detector'
                boxes = self.detector.facebox_detect(img)
            else:
                boxes = self.detector.dapeng_detect(img)
        except Exception, e:
            boxes = []
            with open(self.exist_box)as f:
                dets = f.readlines()
                dets = map(lambda x: map(eval, x.strip().split()), dets)
                for det in dets:
                    x1, y1 = det[0:2]
                    x2, y2 = x1 + det[2], y1 + det[3]
                    box = [x1, x2, y1, y2]
                    boxes.append(box)
        width, height, _ = img.shape
        key_points = []
        for box in boxes:
            x1, x2, y1, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(x2, width), min(y2, height)
            if x2 - x1 <= 0 or y2 - y1 <= 0: continue
            image = cv2.resize(img[y1:y2, x1:x2], (self.IMAGE_SIZE, self.IMAGE_SIZE))
            image = whitering(image)
            w, h = image.shape
            image = nd.array(image).reshape(1, 1, w, h).as_in_context(self.ctx)
            out = self.net(image)
            points = map(lambda x: x / 2.0 + 0.5, out[0].asnumpy())[0].tolist()
            # if not stop:
            #     opt = optimize(points)
            #     new_img = opt.transform(box, img)
            #     if not new_img: break
            #     self.landmark_of_given_img(new_img, use_detector, stop=True)

            cv2.rectangle(img, (x1, y1 + 4), (x2, y2), (255, 0, 0), 2)
            point_pair_l = len(points)

            for i in range(point_pair_l / 2):
                x = points[i] * (x2 - x1) + x1
                y = points[i + point_pair_l / 2] * (y2 - y1) + y1
                # cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 2)
                img[int(y), int(x)] = 255  # [0, 0, 255]
                key_points += [x, y]
        return key_points


# failure rate on valid dataset
def to_evalue(model_path='log/tcdcn.pt', Is_HDF=True):
    batch_size = 256
    if Is_HDF:
        _, _, test_data, _ = HDF5_dataset(batch_size)
    else:
        _, test_data = database(batch_size)
    evalue(model_path, test_data, Is_HDF)


if __name__ == "__main__":
    # images_dirs = ["/home/ldc/work/faceAlignment/face-landmark/image/"]
    images_dirs = ['/home2/block_box/nir_pass/', '/home2/spetrum_box/nir_initial/', '/home2/vis_box/vis/']
    model_path = 'log_DP/tcdcn.pt'
    tm = test_model(model_path)
    for images_dir in images_dirs:
        os.system('rm -f %s' % (images_dir + '*TCNN*'))
        image_list = file_list_fn(images_dir)
        tm.test(image_list, False)
    # to_evalue(Is_HDF=True)
