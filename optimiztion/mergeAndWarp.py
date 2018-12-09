# merge vis and nir;
# detecte face box and face landmark
# transform those point to source image coordinataion
from __init__ import *
import numpy as np


def point_affine(M, keypoints):
    key = np.array(keypoints).reshape(-1, 2)
    M1, M2 = M[:, :2], M[:, -1]
    res = np.array(np.matrix(key - M2) * np.matrix(M1).I)
    res = res.reshape(-1).tolist()
    return res


# coordiante same as the one of target
def align_inst(source, target):
    # Load source image and target image
    # Create instance
    al = Align(source, target, threshold=1)
    # Image transformation
    merge_img, M = al.align_image()
    return merge_img, M


def trans_points(M, target_key, target_box):
    # al.show_with_points(M, target_key, target_box)
    key = point_affine(M, target_key)
    box = point_affine(M, target_box)
    return key, box


if __name__ == "__main__":
    t = "/home/ldc/work/ImageRegistration/Images/27_nir.bmp"
    s = "/home/ldc/work/ImageRegistration/Images/27_vis.bmp"
    nir_key = [342.08, 192.6, 391.09, 187.9, 358.27,
               219.42, 343.53, 248.59, 390.88, 248.09]
    nir_box = [322, 132, 322 + 112, 132 + 144]
    m, M = align_inst(s, t)
    k, b = trans_points(M, nir_key, nir_box)
