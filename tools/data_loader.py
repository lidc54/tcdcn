from mxnet.gluon.data import dataset
from mxnet import nd
import cv2, time, h5py, random, math, sys
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
from detector import Face_Roi
from __init__ import *


class loader(dataset.Dataset):
    def __init__(self, root='/home1/LS3D-W/MTFL/', file='training.txt', detect_fun='facebox', mode='train'):
        # , path, suffix, json_path = None, transform = None
        super(loader, self).__init__()
        self.root = root  # '/home1/LS3D-W/MTFL/'
        self.file = file  # 'training.txt'
        self.mode = mode
        self.read_txt()
        self.IMAGE_SIZE = 40
        self.rt = Rotate_Img()
        self.detector = Face_Roi(fun=detect_fun)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # return:img, class weight, attribute values
        data = self.get_data(self.items[idx])
        return data

    def read_txt(self):
        # self.items = []
        with open(self.root + self.file)as f:
            self.items = f.readlines()

    # ------------- deal with face detector ---------------------------
    def get_data(self, item):
        content = item.strip().split()
        image = self.root + content[0].replace('\\', '/')
        point = map(eval, content[1:11])
        attr = map(lambda x: x - 1, map(eval, content[11:]))
        img, landmark = self.read(image, point)
        return img, landmark, attr

    def read(self, jpg, point):
        img = cv2.imread(jpg)
        if self.mode == 'train':
            img, point = self.rt.rotation(img, point)
        box = self.detector.detect(img, point, self.IMAGE_SIZE)
        box = map(int, box)
        # x1, x2, y1, y2 = box
        # cv2.rectangle(img, (x1, y1 + 4), (x2, y2), (0, 0, 255), 2)
        img, point = self.crop(img, point, box)
        img, point = self.mirror(img, point)
        img, point = self.resize_data(img, point, self.IMAGE_SIZE)
        img = self.whitering(img)
        return img, point

    def crop(self, img, data, box):
        '''crop to box'''
        point = data[:]
        Len = len(data) / 2
        point[0:Len] = map(lambda x: x - box[0], data[0:Len])
        point[Len:] = map(lambda x: x - box[2], data[Len:])
        image = img[box[2]:box[3], box[0]:box[1]]
        return image, point

    def mirror(self, img, points):
        '''mirror to left-right'''
        if random.choice([0, 1]):
            img = img[:, ::-1]
            h, w, c = img.shape
            idx = [1, 0, 2, 4, 3]  # only for five point
            L = len(points) / 2
            resX, resY = [], []
            for i in idx:
                x, y = points[i], points[i + L]
                resX.append(w - x)
                resY.append(y)
            points[:] = resX + resY
        return img, points

    # set landmark to [-1,1]
    def resize_data(self, img, point, IMAGE_SIZE):
        if len(img.shape) == 2:
            height, width = img.shape
        else:
            height, width, _ = img.shape
        Len = len(point) / 2
        point[0:Len] = map(lambda x: (1.0 * x / width - 0.5) * 2.0, point[0:Len])
        point[Len:] = map(lambda x: (1.0 * x / height - 0.5) * 2.0, point[Len:])
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        return img, point

    def whitering(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m, s = cv2.meanStdDev(img)
        img = (img - m) / (1.e-6 + s)
        img = img.astype('float32')
        return img


class Rotate_Img(object):

    def ImgRotate(self, srcImg, degree):
        h, w, c = srcImg.shape
        angle = degree * np.pi / 180.0
        height = max(int(w * abs(math.sin(angle)) + h * abs(math.cos(angle))), h)
        width = max(int(h * abs(math.sin(angle)) + w * abs(math.cos(angle))), w)
        # diaLength = int(math.sqrt(h**2 + w**2))
        tempImg = np.zeros((height, width, c), dtype=srcImg.dtype)
        tx = width / 2 - w / 2  # left
        ty = height / 2 - h / 2  # top
        # print tx ,ty,width,height
        tempImg[ty:ty + h, tx:tx + w] = srcImg
        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        imgRotation = cv2.warpAffine(tempImg, matRotation, (width, height), borderValue=(0, 0, 0))
        return imgRotation

    def getPointAffinedPos(self, x, y, h, w, degree):
        angle = degree * np.pi / 180.0
        height = max(int(w * abs(math.sin(angle)) + h * abs(math.cos(angle))), h)
        width = max(int(h * abs(math.sin(angle)) + w * abs(math.cos(angle))), w)
        diaLength = math.sqrt(h ** 2 + w ** 2)
        center_x, center_y = width / 2, height / 2
        x -= w / 2
        y -= h / 2

        dst_x = round(x * math.cos(angle) + y * math.sin(angle) + center_x)
        dst_y = round(-x * math.sin(angle) + y * math.cos(angle) + center_y)
        return dst_x, dst_y

    def rotation(self, img, point, degree=None):
        boundary = 15  # degree 15/180:pi
        if not degree: degree = random.randrange(-boundary, boundary, 2)
        imgrot = self.ImgRotate(img, degree)
        h, w, _ = img.shape

        resX, resY = [], []
        L = len(point) / 2
        for i in range(L):
            x, y = point[i], point[i + L]
            rx, ry = self.getPointAffinedPos(x, y, h, w, degree)
            resX.append(rx)
            resY.append(ry)
        res = resX + resY
        return imgrot, res


class watch_image(object):

    # for show
    def showimg(self, imgs, points, path=''):
        img = self.Detrans(imgs)
        try:
            h, w = img.shape
        except Exception, e:
            h, w, _ = img.shape
        point_pair_l = len(points)
        for i in range(point_pair_l / 2):
            if 0 <= points[i] <= 1.0:
                x = points[i] * w  # * (x2 - x1) + x1
                y = points[i + point_pair_l / 2] * h  # * (y2 - y1) + y1
            else:
                x = points[i]  # * (x2 - x1) + x1
                y = points[i + point_pair_l / 2]  # * (y2 - y1) + y1
            # print 'X:',int(x),'Y:',int(y)
            cv2.circle(img, (int(x), int(y)), 1, (255, 255, 0), 2)
            # img[int(y), int(x)]=255
        saved = True
        if path:
            path = 'result/' + path.split('/')[-1]
            # print path
            saved = cv2.imwrite(path, img)
        if not saved or not path:
            plt.imshow(img)
            plt.show()

    # for show
    def Detrans(self, imgs):
        img = imgs * 255.0 / (np.max(imgs) - np.min(imgs))
        img[img > 255] = 255
        img[img < 0] = 0
        img = img.astype(np.uint8)
        return img


# ~~~~~~~~~~~~~~~~~~~~~~~~~data_loader~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def bachfy(datasets):
    img, landmark, attr = zip(*datasets)
    img = nd.array(img)
    img = nd.expand_dims(img, axis=1)
    landmark = nd.array(landmark)
    attr = nd.array(attr)
    return img, landmark, attr


def database(batch_size):
    root = '/home1/LS3D-W/MTFL/'
    train_file = 'training.txt'
    test_file = 'testing.txt'
    numworks = 0
    train_data = mx.gluon.data.DataLoader(loader(root, train_file), batch_size, last_batch='discard',
                                          num_workers=numworks, batchify_fn=bachfy, shuffle=True)
    test_data = mx.gluon.data.DataLoader(loader(root, test_file), batch_size, last_batch='discard',
                                         num_workers=numworks, batchify_fn=bachfy, shuffle=True)
    return train_data, test_data


# ~~~~~~~~~~~~~~~~~~~~~~~~~a model for data_loader~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_bachfy(dataset):
    img, landmark, attr = zip(*dataset)
    return img[0], landmark[0], attr[0]


def test_iter():
    batch_size = 128
    numworks = 0
    detector = ['facebox', 'dlib', 'dapeng']
    dete_fun = detector[0]
    if dete_fun == detector[0]:
        numworks = 0
    # todo:error occurs when  using mp in pytorch:
    # https://github.com/pytorch/pytorch/issues/2517
    monitor = watch_image()
    # root = '/home1/LS3D-W/MTFL/', file = 'training.txt'
    train_data = mx.gluon.data.DataLoader(loader(detect_fun=dete_fun),
                                          batch_size, last_batch='discard', num_workers=numworks,
                                          batchify_fn=test_bachfy, shuffle=True)
    j = 0
    t = time.time()
    for i, data in enumerate(train_data):
        print i, time.time() - t
        t = time.time()
        pad = random.randint(10, 500)
        if i % pad != 0: continue
        img, point, attr = data
        # refer line 91
        point = map(lambda x: x / 2.0 + 0.5, point)
        name = repr(j) + '_'.join(map(repr, attr)) + '.jpg'
        print name
        monitor.showimg(img, point)  # , name
        j += 1
        if j > 10: break


# ~~~~~~~~~~~~~~~~~~~~~~~~~a model for hdf5_loader~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_HDF5_loader():
    file = '/home1/LS3D-W/MTFL/X40/_X40_trainHDF.h5'
    attr = ["landmarks", "gender", "smile", "glasses", "pose"]
    file = 'X40_trainHDF.h5'
    monitor = watch_image()
    # attr = ["landmarks", 'attrs']
    # labels = []
    # with h5py.File(file) as f:
    #     data = f['X'][:]
    #     for a in attr:
    #         labels.append(f[a][:])
    # L, _, _, _ = data.shape
    # dataiter = mx.io.NDArrayIter(data, labels, 1, True, last_batch_handle='discard')
    dataiter, L = HDF5_Loader(file, 1, 'test')
    j = 0

    for i, batch in enumerate(dataiter):
        pad = random.randint(1, max(L / 10, 1))
        if i % pad != 0: continue
        img = batch.data[0][0, 0].asnumpy()
        landmark = batch.data[1][0].asnumpy().tolist()
        landmark = map(lambda x: x / 2.0 + 0.5, landmark)
        print map(lambda x: x[0].asnumpy().tolist(), batch.data[2:]),
        name = repr(j) + '.jpg'
        print name
        monitor.showimg(img, landmark, name)
        j += 1
        if j > 10: break
        # dataiter.reset()


def HDF5_Loader(file, batch_size, mode='train', shuffle=True):
    attr = ["landmarks", 'attrs']
    # labels = []
    data = []
    with h5py.File(file) as f:
        data.append(f['X'][:])
        for a in attr:
            data.append(f[a][:])
    L, _, _, _ = data[0].shape
    last_batch = 'roll_over'
    if mode != 'train': last_batch = 'discard'
    data_iter = mx.io.NDArrayIter(data, label=None, batch_size=batch_size, shuffle=shuffle,
                                  last_batch_handle=last_batch)
    return data_iter, L


def HDF5_dataset(batch_size):
    train_file = 'X40_trainHDF.h5'
    test_file = 'X40_testHDF.h5'
    train_data, L_train = HDF5_Loader(train_file, batch_size=batch_size)
    test_data, L_test = HDF5_Loader(test_file, batch_size=batch_size)
    return train_data, L_train, test_data, L_test


def HDF5_Creator():
    batch_size = 8
    numworks = 2
    IMAGE_SIZE = 40
    _iters = 50  # the numbers of iteration of input data
    detector = ['facebox', 'dlib', 'dapeng']
    dete_fun = detector[2]  # fixed
    root = '/home1/LS3D-W/MTFL/'
    train_file = 'training.txt'
    test_file = 'testing.txt'
    suffix = 'X40_'

    train_data = mx.gluon.data.DataLoader(loader(root, train_file, detect_fun=dete_fun), batch_size,
                                          last_batch='discard', num_workers=numworks,
                                          batchify_fn=bachfy, shuffle=True)
    test_data = mx.gluon.data.DataLoader(loader(root, test_file, detect_fun=dete_fun), batch_size,
                                         last_batch='discard', num_workers=numworks,
                                         batchify_fn=bachfy, shuffle=True)
    train_HDF_path = suffix + train_file.split('ing')[0] + 'HDF.h5'
    test_HDF_path = suffix + test_file.split('ing')[0] + 'HDF.h5'

    # attributes = ["landmarks", "gender", "smile", "glasses", "pose"]
    t = time.time()
    with h5py.File(train_HDF_path, 'w')as f:
        entries = batch_size * len(train_data)
        _SIZE = entries * _iters
        HDF5_imgs = f.create_dataset("X", shape=(_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE), dtype='float32')
        HDF5_keypoints = f.create_dataset("landmarks", shape=(_SIZE, 10), dtype='float32')
        HDF5_attrs = f.create_dataset("attrs", shape=(_SIZE, 4), dtype='float32')
        for iter in range(_iters):
            base = iter * entries
            for i, data in enumerate(train_data):
                img, point, attr = data
                start, end = i * batch_size + base, (i + 1) * batch_size + base
                HDF5_imgs[start:end] = img.asnumpy()
                HDF5_keypoints[start:end] = point.asnumpy()
                HDF5_attrs[start:end] = attr.asnumpy()

            print iter, 'train data created!', time.time() - t
            t = time.time()

    with h5py.File(test_HDF_path, 'w')as f:
        _SIZE = batch_size * len(test_data)
        HDF5_imgs = f.create_dataset("X", shape=(_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE), dtype='float32')
        HDF5_keypoints = f.create_dataset("landmarks", shape=(_SIZE, 10), dtype='float32')
        HDF5_attrs = f.create_dataset("attrs", shape=(_SIZE, 4), dtype='float32')
        for i, data in enumerate(test_data):
            img, point, attr = data
            start, end = i * batch_size, (i + 1) * batch_size
            HDF5_imgs[start:end] = img.asnumpy()
            HDF5_keypoints[start:end] = point.asnumpy()
            HDF5_attrs[start:end] = attr.asnumpy()

        print 'test data created!', time.time() - t
        t = time.time()


def create_hdf_with_various_jitter():
    HDF5_Creator()
    test_HDF5_loader()


if __name__ == "__main__":
    # test_iter()
    test_HDF5_loader()
