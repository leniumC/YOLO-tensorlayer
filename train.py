from model import get_model
from utils import read_data
import tensorlayer as tl
import tensorflow as tf


def main():
    w_1 = 1
    w_2 = 1
    lr = 0.001
    n_epochs = 1000
    n_batches = 10

    X = tf.placeholder(tf.float32, [None, 256, 256, 3])
    y_ = tf.placeholder(tf.float32, [None, 16 * 16 * 5])

    net_train = get_model(X, is_train=True, reuse=False)
    net_val = get_model(X, is_train=False, reuse=True)

    y_train = net_train.outputs
    y_val = net_val.outputs

    loss_train = w_1 * tf.losses.mean_squared_error(y_, y_train) + w_2 * tf.losses.sigmoid_cross_entropy(y_, y_train)
    loss_val = w_1 * tf.losses.mean_squared_error(y_, y_val) + w_2 * tf.losses.sigmoid_cross_entropy(y_, y_val)

    op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_train)

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    for it in range(n_epochs):
        for num in range(n_batches):
            imgs, labels = read_data(num, n_batches)
            print(labels.shape)
            feed_dict = {X: imgs, y_: labels}
            _, loss = sess.run([op, loss_train], feed_dict=feed_dict)
            print(loss)


if __name__ == '__main__':
    main()