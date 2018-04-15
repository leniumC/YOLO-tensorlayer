import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import *


def conv(input, id, filters, size, stride):
    return Conv2dLayer(
        input,
        act=tl.activation.leaky_relu,
        shape=[size, size, input.outputs.shape[3], filters],
        strides=(1, stride, stride, 1),
        padding='SAME',
        name='conv_%d' % id
    )


def pool(input, id, size, stride):
    return PoolLayer(
        input,
        ksize=(1, size, size, 1),
        strides=(1, stride, stride, 1),
        padding='SAME',
        pool=tf.nn.max_pool,
        name='pool_%d' % id
    )


def dense(input, id, n_units, act=tf.identity, flatten=False):
    if flatten:
        flat = FlattenLayer(input, name='flatten_%d' % id)
    else:
        flat = input
    return DenseLayer(
        flat,
        n_units=n_units,
        act=act,
        name='dense_%d' % id
    )


def get_model(X, is_train=True, reuse=False):
    with tf.variable_scope('yolo_model', reuse=reuse):
        set_name_reuse(reuse)

        network = InputLayer(X, name='input')

        network = conv(network, 0, 64, 7, 1)
        network = pool(network, 0, 2, 2)
        network = conv(network, 1, 192, 3, 1)
        network = pool(network, 1, 2, 2)
        network = conv(network, 2, 128, 1, 1)
        network = conv(network, 3, 256, 3, 1)
        network = conv(network, 4, 256, 1, 1)
        network = conv(network, 5, 512, 3, 1)
        network = pool(network, 2, 2, 2)
        network = conv(network, 6, 256, 3, 1)
        network = conv(network, 7, 512, 3, 1)
        network = conv(network, 8, 256, 3, 1)
        network = conv(network, 9, 512, 3, 1)
        network = conv(network, 10, 512, 1, 1)
        network = conv(network, 11, 1024, 3, 1)
        network = pool(network, 3, 2, 2)
        network = conv(network, 12, 512, 1, 1)
        network = conv(network, 13, 1024, 3, 1)
        network = pool(network, 4, 2, 2)
        network = conv(network, 14, 1024, 3, 1)
        network = conv(network, 15, 1024, 3, 1)
        network = pool(network, 5, 2, 2)
        print(network.outputs)

        network = dense(network, 0, 1024, flatten=True, act=tl.activation.leaky_relu)
        network = dense(network, 1, 4096, flatten=False, act=tl.activation.leaky_relu)
        network = dense(network, 2, 32 * 32 * 5, flatten=False, act=tf.identity)

        network = ReshapeLayer(
            network,
            (-1, 32, 32, 5),
            name='reshape'
        )

    return network


def main():
    X = tf.placeholder(tf.float32, [None, 256, 256, 3])
    model = get_model(X)

    model.print_layers()
    model.print_params(False)

    print(model.outputs)


if __name__ == '__main__':
    main()
