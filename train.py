from tools.net import TCDCN, loss_FLD, initia_Tcdcn, L2_penalty
from tools.data_loader import database, HDF5_dataset
from mxboard import SummaryWriter
import mxnet as mx
from mxnet import gluon, autograd, nd
from tools.unity import recoder, ShowNet_PerEpoch, Iface, ttmp
import time


def train():
    net = TCDCN()
    epoches = 1000
    path = 'log_DP/log6/'
    # if os.path.exists(path):
    #     os.system('rm -r %s' % (path))
    sv_model = path + 'tcdcn.pt'
    sw = SummaryWriter(logdir=path)
    ttmp.set(sw)
    # ctx = [mx.gpu(i) for i in [3, 4]]
    ctx = mx.gpu(1)

    initia_Tcdcn(net, ctx)
    # net.load_parameters(sv_model, ctx=ctx)
    lr = 0.001
    batch_size = 256 * len(ctx) if type(ctx) == list else 512
    idx_span = 50
    eta_weight = 0.01
    print '~' * 10, '\npath:', path, '\n', 'learning rate:', lr, '\n', 'ctx:', ctx, '\n', '~' * 10, '\n'
    keypoint_weight = [0.5, 1.5, 1.5, 2, 1]  # "landmarks", "smile", "glasses", "gender", "pose"
    # train_data, test_data = database(batch_size)
    train_data, _, test_data, _ = HDF5_dataset(batch_size)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    items = ["landmarks", "smile", "glasses", "gender", "pose"]

    j, k, epoch = 0, 0, 0
    train_items, test_items = [], []
    while epoch < epoches:
        batch = 0
        ttmp.setstop(True)
        for batch in train_data:
            t = time.time()
            ttmp.setiter(j)
            batch = batch.data
            if type(ctx) == list:
                data = gluon.utils.split_and_load(batch[0], ctx)
                landmark = gluon.utils.split_and_load(batch[1], ctx)
                attr = gluon.utils.split_and_load(batch[2], ctx)
                with autograd.record():
                    out = [net(X) for X in data]
                    loss_items = [loss_FLD(X, Y, Z, keypoint_weight) for X, Y, Z in zip(out, landmark, attr)]
                    L2_weight = L2_penalty(net, eta_weight)
                    losses = [nd.sum(nd.stack(*X)) + Y[0] for X, Y in zip(loss_items, L2_weight)]
                for loss in losses:
                    loss.backward()
            else:
                data = batch[0].as_in_context(ctx)
                landmark = batch[1].as_in_context(ctx)
                attr = batch[2].as_in_context(ctx)
                with autograd.record():
                    out = net(data)
                    loss_items = loss_FLD(out, landmark, attr, keypoint_weight)
                    L2_weight = L2_penalty(net, eta_weight)
                    losses = nd.sum(nd.stack(*loss_items)) + L2_weight[0]
                losses.backward()
            trainer.step(batch_size)
            if type(ctx) == list:
                loss_items = [nd.sum(nd.stack(*X)) for X in zip(*loss_items)]
            for item_idx, tag in enumerate(items):
                value = nd.sum(loss_items[item_idx]).asscalar() / batch_size
                sw.add_scalar(tag, value=value, global_step=j)
            recoder(loss_items, sw, j, train_items, test_items, mode='train', span=idx_span)
            j += 1
            if j % 20 == 0: print epoch, 'train ok%2.5s' % (time.time() - t)
            # if j > 4: break
        ttmp.setstop(False)
        ShowNet_PerEpoch(net, sw, batch[1], epoch)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for batch in test_data:
            batch = batch.data
            t = time.time()
            if type(ctx) == list:
                data = gluon.utils.split_and_load(batch[0], ctx)
                landmark = gluon.utils.split_and_load(batch[1], ctx)
                attr = gluon.utils.split_and_load(batch[2], ctx)
                out = [net(X) for X in data]
                loss_items = [loss_FLD(X, Y, Z, keypoint_weight) for X, Y, Z in zip(out, landmark, attr)]
                loss_items = [nd.sum(nd.stack(*X)) for X in zip(*loss_items)]
            else:
                data = batch[0].as_in_context(ctx)
                landmark = batch[1].as_in_context(ctx)
                attr = batch[2].as_in_context(ctx)
                out = net(data)
                loss_items = loss_FLD(out, landmark, attr, keypoint_weight)
            for item_idx, tag in enumerate(items):
                value = nd.sum(loss_items[item_idx]).asscalar() / batch_size
                sw.add_scalar(tag + '_test', value=value, global_step=k)
            recoder(loss_items, sw, k, train_items, test_items, mode='test', span=idx_span)
            k += 1
            if k % 20 == 0: print epoch, 'test ok%2.5s' % (time.time() - t)
            # if k > 142:
            #     Iface.idx["glasses"] = 0
            # if k > 198:
            #     Iface.idx["gender"] = 0
            # if k > 195:
            #     Iface.idx["smile"] = 0
            # if k > 108:
            #     Iface.idx["pose"] = 0
            # if k > 4: break
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        epoch += 1
        print 'save model'
        net.save_parameters(sv_model)
        train_data.reset()
        test_data.reset()
        # break


if __name__ == "__main__":
    train()
