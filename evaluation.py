import os, cv2, h5py
from torch.utils.serialization import load_lua
from tools.data_loader import watch_image
from test import evalue
from tools.data_loader import loader, bachfy, HDF5_Loader
import mxnet as mx


def get_five_points_from_68(points):
    five_points = [36, 39, 42, 45, 30, 48, 54]
    res = []
    for i in five_points:
        res += points[i]
    tmp = []
    for i in [0, 1, 4, 5]:
        tmp.append((res[i] + res[i + 2]) / 2)
    tmp += res[8:]
    x, y = tmp[::2], tmp[1::2]
    return x + y


def search_file(path):
    base = ['png', 'PNG', 'bmp', 'jpg']
    for root, dirs, files in os.walk(path):
        for file in files:
            if not sum(map(lambda x: x in file, base)):
                continue
            jpg = os.path.join(root, file)
            # img = cv2.imread(jpg)
            t7file = '.'.join(jpg.split('.')[:-1] + ['t7'])
            points = load_lua(t7file, long_size=8)
            five_points = get_five_points_from_68(points.tolist())
            yield file, five_points


def create_AFLW_HDF5():
    path = "/home1/LS3D-W/AFLW2000-3D-Reannotated/"
    res = []
    batch_size = 8
    numworks = 2
    IMAGE_SIZE = 40
    dete_fun = 'dapeng'
    suffix = 'X40_'

    for jpg, point in search_file(path):
        out = [jpg] + map(repr, point) + ['0'] * 4 + ['\n']
        out = ' '.join(out)
        res.append(out)
    with open(path + 'list.txt', 'w')as f:
        f.writelines(''.join(res))
    test_data = mx.gluon.data.DataLoader(loader(path, 'list.txt', detect_fun=dete_fun), batch_size,
                                         last_batch='discard', num_workers=numworks,
                                         batchify_fn=bachfy, shuffle=True)
    h5 = path + suffix + 'eval.h5'
    with h5py.File(h5, 'w')as f:
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
    return h5


if __name__ == "__main__":
    '''create AFLW as HDF5 files for evalution
        hdf5 was selected for evalution with dapeng detection method.
        another is selected for facebox detection method'''
    # test_file = create_AFLW_HDF5() # the file has been created, and disabled this function
    # print test_file
    Is_hdf = True
    batch_size = 256
    if Is_hdf:
        test_file = '/home1/LS3D-W/AFLW2000-3D-Reannotated/X40_eval.h5'
        test_data, L_test = HDF5_Loader(test_file, batch_size=batch_size)
    else:
        root = '/home1/LS3D-W/AFLW2000-3D-Reannotated/'
        test_file = 'list.txt'
        test_data = mx.gluon.data.DataLoader(loader(root, test_file), batch_size, last_batch='discard',
                                             num_workers=0, batchify_fn=bachfy, shuffle=False)
    model_path = 'log_DP/log6/tcdcn.pt'
    evalue(model_path, test_data, Is_hdf)
