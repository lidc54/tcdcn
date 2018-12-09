from mxnet.gluon import nn, HybridBlock
from mxnet import nd, gluon
import mxnet as mx
import mxnet.gluon.loss as loss
from unity import Iface, ttmp
import numpy as np


class Abs(HybridBlock):
    def __init__(self, **kwargs):
        super(Abs, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return nd.abs(x)
        # return super(Abs, self).hybrid_forward(F, x, *args, **kwargs)


class TCDCN(HybridBlock):
    def __init__(self, **kwargs):
        super(TCDCN, self).__init__(**kwargs)
        self.block()
        self.multi_loss()

    def hybrid_forward(self, F, x, *args, **kwargs):
        # feature = self.base(x)
        for f in self.base:
            x = f(x)
        feature = x
        # "landmarks", "smile", "glasses", "gender", "pose"
        return self.landmark(feature), \
               Iface.idx["smile"] * self.smile(feature), \
               Iface.idx["glasses"] * self.glasses(feature), \
               Iface.idx["gender"] * self.gender(feature), \
               Iface.idx["pose"] * self.pose(feature)

    def multi_loss(self):
        self.landmark = nn.Dense(units=10, in_units=100, prefix='dense1_landmark')
        self.gender = nn.Dense(units=2, in_units=100, prefix='dense2_genderNet')
        self.glasses = nn.Dense(units=2, in_units=100, prefix='dense3_glassesNet')
        self.smile = nn.Dense(units=2, in_units=100, prefix='dense4_smileNet')
        self.pose = nn.Dense(units=5, in_units=100, prefix='dense5_poseNet')

    def block(self):
        self.base = nn.HybridSequential()
        # channel, kernel_size, pad, pooling_size, pooling_stride
        # architecture = ((16, 5, 2, 2, 2), (48, 3, 1, 2, 2), (64, 3, 0, 3, 2), (64, 2, 0))
        architecture = ((16, 5, 0, 2, 2), (48, 3, 0, 2, 2), (64, 3, 0, 3, 2), (64, 2, 0))
        in_channels = 1
        for arch in architecture:
            if len(arch) == 5:
                channels, kernel_size, pad, pk, ps = arch
            else:
                channels, kernel_size, pad = arch
                pk, ps = 0, 0
            conv = nn.Conv2D(channels, in_channels=in_channels, strides=1,
                             kernel_size=kernel_size, padding=pad)
            in_channels = channels
            activ = nn.Activation('tanh')
            abs1 = Abs()
            self.base.add(conv, activ, abs1)
            if pk:
                pool = nn.MaxPool2D(pool_size=pk, strides=ps, ceil_mode=True)
                self.base.add(pool)
        self.base.add(nn.Dense(units=100, prefix='dense0_bone'))  # in_units=576,
        self.base.add(nn.Activation('tanh'), Abs())


def get_params(net):
    params = {}
    for k, v in net.base.collect_params().items():
        params[k] = v
    key_data = ["landmarks", "smile", "glasses", "gender", "pose"]
    for k in net.keys():
        if sum([item in k for item in key_data]) > 0:
            for k_item, v_item in net[k].collect_params().items():
                params[k_item] = v_item
    return params


def initia_Tcdcn(net, ctx):
    for k, v in net.collect_params().items():
        if 'bias' in k:
            v.initialize(mx.initializer.Constant(0.0), ctx=ctx)
        else:
            v.initialize(mx.initializer.Xavier(magnitude=3), ctx=ctx)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def extraMSE(y_pred, y_true):
    num, L = y_true.shape
    extra_eye_weight = 100.0
    extra_side_weight = 1.0
    # eye loss
    eyeX_pred = y_pred[:, 0] - y_pred[:, 1]  # del X size 16
    eyeY_pred = y_pred[:, 5] - y_pred[:, 6]
    eyeX_true = y_true[:, 0] - y_true[:, 1]
    eyeY_true = y_true[:, 5] - y_true[:, 6]
    eye_diff = eyeY_pred / (1e-6 + eyeX_pred) - eyeY_true / (1e-6 + eyeX_true)
    eye_loss = nd.sum(eye_diff ** 2) / (num * 2.0)
    # side face

    # L /= 2
    # # delX = y_true[:, 0] - y_true[:, 1]  # del X size 16
    # # delY = y_true[:, L] - y_true[:, L + 1]  # del y size 16
    # interOc = (1e-6 + (eyeX_true ** 2 + eyeY_true ** 2) ** 0.5)  # .T
    # left_sideX = y_pred[:, 0] - y_pred[:, 2]
    # left_sideX_true = y_true[:, 0] - y_true[:, 2]
    # right_sideX = y_pred[:, 1] - y_pred[:, 2]
    # right_sideX_true = y_true[:, 1] - y_true[:, 2]
    # side_diff = (left_sideX - left_sideX_true) ** 2 + \
    #             (right_sideX - right_sideX_true) ** 2
    # side_diff = side_diff / (interOc ** 2)
    # side_loss = nd.sum(side_diff) / 2.0
    if ttmp.do:
        # ttmp.sw.add_scalar('side_loss', value=side_loss.asscalar(), global_step=ttmp.iter)
        ttmp.sw.add_scalar('eye_loss', value=eye_loss.asscalar(), global_step=ttmp.iter)

    return eye_loss * extra_eye_weight  # + side_loss * extra_side_weight


def left_right_weight(y_true):
    num, L = y_true.shape
    L /= 2
    delX = y_true[:, 0] - y_true[:, 1]  # del X size 16
    # delY = y_true[:, L] - y_true[:, L + 1]  # del y size 16
    # interOc = (1e-6 + (delX * delX + delY * delY) ** 0.5).T
    left = delX / (1e-6 + y_true[:, 0] - y_true[:, 2])
    right = delX / (1e-6 + y_true[:, 1] - y_true[:, 2])
    weight = np.vstack((left, right, np.ones_like(left), left, right))
    weight = np.tile(weight, (2, 1))
    weight = np.abs(weight)
    weight[weight > 10] = 10
    return weight


def NormlizedMSE(y_pred, y_true):
    num, L = y_true.shape
    L /= 2
    delX = y_true[:, 0] - y_true[:, 1]  # del X size 16
    delY = y_true[:, L] - y_true[:, L + 1]  # del y size 16
    interOc = (1e-6 + (delX * delX + delY * delY) ** 0.5).T
    # Cannot multiply shape (16,10) by (16,1) so we transpose to (10,16) and (1,16)
    diff = (y_pred - y_true).T  # Transpose so we can divide a (16,10) array by (16,1)
    diff = (diff / interOc).T  # We transpose back to (16,10)
    weight = left_right_weight(y_true.asnumpy())
    weight = nd.array(weight.T, ctx=y_true.context)
    diff = diff * weight
    resLoss = nd.sum(diff ** 2) / (num * 2.0)  # Loss is scalar
    if ttmp.do:
        ttmp.sw.add_scalar('res_loss', value=resLoss.asscalar(), global_step=ttmp.iter)

    extraLoss = extraMSE(y_pred, y_true)
    return resLoss + extraLoss


def loss_FLD(pre, label, attr, keypoint_weight):
    L_weight, S_weight, GL_weight, GE_weight, P_weight = keypoint_weight
    # "landmarks", "smile", "glasses", "gender", "pose"
    sigmoid = loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    softmax = loss.SoftmaxCrossEntropyLoss(from_logits=False, sparse_label=False)  # sparse_label=True
    L, others = attr.shape
    ctx = attr.context
    res_attr, dims = [], [2, 2, 2, 5]
    for t in range(others):
        tmp = nd.zeros((L, dims[t])).as_in_context(ctx)
        tmp[nd.arange(L).as_in_context(ctx), attr[:, t]] = 1
        res_attr.append(tmp)
    smile, glasses, gender, pose = res_attr
    Plandmarks, Psmile, Pglasses, Pgender, Ppose = pre
    Lloss = NormlizedMSE(Plandmarks, label) * L_weight
    Sloss = Iface.idx["smile"] * sigmoid(Psmile, smile) * S_weight
    GLloss = Iface.idx["glasses"] * sigmoid(Pglasses, glasses) * GL_weight
    GEloss = Iface.idx["gender"] * sigmoid(Pgender, gender) * GE_weight
    Ploss = Iface.idx["pose"] * softmax(Ppose, pose) * P_weight
    return Lloss, nd.sum(Sloss), nd.sum(GLloss), nd.sum(GEloss), nd.sum(Ploss)


def L2_penalty(net, eta_weight):
    d = ['conv0_weight', 'conv0_bias', 'conv1_weight', 'conv1_bias', 'conv2_weight', 'conv2_bias', 'conv3_weight',
         'conv3_bias', 'dense0_boneweight', 'dense0_bonebias', 'dense1_landmarkweight', 'dense1_landmarkbias',
         'dense2_genderNetweight', 'dense2_genderNetbias', 'dense3_glassesNetweight', 'dense3_glassesNetbias',
         'dense4_smileNetweight', 'dense4_smileNetbias', 'dense5_poseNetweight', 'dense5_poseNetbias']
    solver = net.collect_params()
    res = []

    for k, v in solver.items():
        if 'bias' in k: continue
        flag = False
        for key in Iface.idx.keys():
            if Iface.idx[key] == 0 and key in k:
                flag = True
                break
        if flag: continue
        if len(v.list_ctx()) > 1:
            tmp = []
            for i in range(len(v.list_ctx())):
                tmp.append(v.list_data()[i])
            res.append(tmp)
        else:
            res.append(v.data())

    losses = []
    if type(res[0]) == list:
        for r in zip(*res):
            tmp = []
            for item in r:
                tmp.append(nd.sum(item ** 2) / 2)
            losses.append(nd.sum(nd.stack(*tmp)) * eta_weight)
    else:
        tmp = []
        for item in res:
            tmp.append(nd.sum(item ** 2) / 2)
        losses.append(nd.sum(nd.stack(*tmp)) * eta_weight)
    return losses


if __name__ == "__main__":
    pass
