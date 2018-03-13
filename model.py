import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import *


def conv(input, id, filters, size, stride):
    return Conv2dLayer(
        input,
        act=tl.activation.leaky_relu,
        shape=(size, size, input.outputs.shape[3], filters),
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

        network = conv(network, 0, 16, 3, 1)
        network = pool(network, 0, 2, 2)
        network = conv(network, 1, 32, 3, 1)
        network = pool(network, 1, 2, 2)
        network = conv(network, 2, 64, 3, 1)
        network = pool(network, 2, 2, 2)
        network = conv(network, 3, 128, 3, 1)
        network = pool(network, 3, 2, 2)
        network = conv(network, 4, 256, 3, 1)
        network = pool(network, 4, 2, 2)
        network = conv(network, 5, 512, 3, 1)
        network = pool(network, 5, 2, 2)

        network = conv(network, 6, 1024, 3, 1)
        network = conv(network, 7, 1024, 3, 1)
        network = conv(network, 8, 1024, 3, 1)

        network = dense(network, 0, 256, flatten=True, act=tl.activation.leaky_relu)
        network = dense(network, 1, 4096, flatten=False, act=tl.activation.leaky_relu)
        network = dense(network, 2, 16 * 16 * 5, flatten=False, act=tf.identity)

    return network


def main():
    X = tf.placeholder(tf.float32, [None, 256, 256, 3])
    model = get_model(X)

    model.print_layers()
    model.print_params(False)


if __name__ == '__main__':
    main()
