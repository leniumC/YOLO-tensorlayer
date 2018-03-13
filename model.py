import tensorlayer as tl
from tensorlayer.layers import *


def Conv(input, id, filters, size, stride):
    return Conv2dLayer(
        input,
        act=tl.activation.leaky_relu,
        shape=(size, size, input.outputs.shape[3], filters),
        strides=(1, stride, stride, 1),
        padding='SAME',
        name='conv_%d' % id
    )


def Pool(input, id, size, stride):
    return PoolLayer(
        input,
        ksize=(1, size, size, 1),
        strides=(1, stride, stride, 1),
        padding='SAME',
        pool=tf.nn.max_pool,
        name='pool_%d' % id
    )


def get_model(X, is_train=True, reuse=False):
    with tf.variable_scope('yolo_model', reuse=reuse):
        set_name_reuse(reuse)

        net = InputLayer(X, name='input')
        net = Conv(net, 0, 16, 3, 1)
        net = Pool(net, 0, 2, 2)
        net = Conv(net, 1, 32, 3, 1)
        net = Pool(net, 1, 2, 2)
        net = Conv(net, 2, 64, 3, 1)
        net = Pool(net, 2, 2, 2)
        net = Conv(net, 3, 128, 3, 1)
        net = Pool(net, 3, 2, 2)
        net = Conv(net, 4, 256, 3, 1)
        net = Pool(net, 4, 2, 2)
        net = Conv(net, 5, 512, 3, 1)
        net = Pool(net, 5, 2, 2)
        net = Conv(net, 6, 1024, 3, 1)
        net = Conv(net, 7, 1024, 3, 1)
        net = Conv(net, 8, 1024, 3, 1)

        # TODO: complete this

        return net


def main():
    X = tf.placeholder(tf.float32, [None, 256, 256, 3])
    model = get_model(X)

    model.print_layers()
    model.print_params(False)


if __name__ == '__main__':
    main()
