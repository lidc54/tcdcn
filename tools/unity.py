from mxnet import nd
import numpy as np


class tmp(object):
    def set(self, sw):
        self.sw = sw

    def setiter(self, iter):
        self.iter = iter

    def setstop(self, stop):
        self.do = stop


ttmp = tmp()


class I(object):
    def __init__(self):
        self.idx = {"smile": 1, "glasses": 1, "gender": 1, "pose": 1}

    def set(self, key, ctx):
        self.idx[key] = 0


def DtransImage(img):
    sz = img.shape
    mmin, mmax = 0, 1
    if len(sz) == 2:
        mmin = nd.min(img, keepdims=True)
        mmax = nd.max(img, keepdims=True)
    elif len(sz) == 3:
        mmin = nd.min(img, axis=(1, 2), keepdims=True)
        mmax = nd.max(img, axis=(1, 2), keepdims=True)
    elif len(sz) == 4:
        mmin = nd.min(img, axis=(2, 3), keepdims=True)
        mmax = nd.max(img, axis=(2, 3), keepdims=True)
    # print 'mmin shape:', mmin.shape,'img shape:',img.shape
    imgs = (img - mmin) / (mmax - mmin)
    return imgs


def ShowNet_PerEpoch(net, sw, landmark, epoch):
    solver = net.collect_params()
    for k, v in solver.items():
        if 'bias' in k: continue
        if type(v.list_ctx()) == list:
            sw.add_histogram(tag=k, values=v.list_data()[0].asnumpy(), global_step=epoch, bins=1000)
        else:
            sw.add_histogram(tag=k, values=v.data().asnumpy(), global_step=epoch, bins=1000)
        # if 'conv' in k:
        #     tmp = DtransImage(v.data()[0])
        #     print tmp.shape
        #     sw.add_image(k + '_image', tmp, global_step=epoch)
    sw.add_histogram(tag='landmark', values=landmark, global_step=epoch, bins=1000)


def crita(train_items, test_items, beta=1):
    train = np.array(train_items)
    test = np.array(test_items)
    L, _ = train.shape
    median = np.median(train, axis=0)
    sums = np.sum(train, axis=0)
    min2 = np.min(test, axis=0)

    pre = L * median / (1e-6 + sums - L * median)
    suf = (test_items[-1] - min2) / (1e-6 + beta * min2)
    res = pre * suf
    return res


def recoder(losses, sw, steps, train_items, test_items, mode='train', span=200):
    title = ["landmarks", "smile", "glasses", "gender", "pose"]
    items = nd.stack(*losses[1:]).reshape(-1).asnumpy()
    if mode == 'train':
        train_items.append(items)
        if len(train_items) > span:
            train_items[:] = train_items[1:]
    else:
        test_items.append(items)
        if len(test_items) > span:
            test_items[:] = test_items[1:]
    if mode == 'test' and len(test_items) >= span and len(train_items) >= span:
        v = crita(train_items, test_items)
        for i, t in enumerate(title[1:]):
            # if v[i] > 500: Iface.idx[t] = 0
            sw.add_scalar(t + '_index', value=v[i], global_step=steps)


Iface = I()
