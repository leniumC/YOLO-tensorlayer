from model import get_model
from utils import read_data, loss_function, show_bbox, sigmoid
import tensorlayer as tl
import tensorflow as tf
import os


def main():
    lr = 1e-6
    n_epochs = 200
    n_batches = 200

    X = tf.placeholder(tf.float32, [None, 256, 256, 3])
    y_ = tf.placeholder(tf.float32, [None, 32, 32, 5])

    net_train = get_model(X, is_train=True, reuse=False)
    # net_val = get_model(X, is_train=False, reuse=True)

    y_train = net_train.outputs
    # y_val = net_val.outputs
    net_train.print_layers()
    net_train.print_params(False)

    loss_train, coord_loss, len_loss, is_obj_loss, no_obj_loss = loss_function(y_, y_train)
    # loss_val = loss_function(y_, y_val)

    # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # gvs = optimizer.compute_gradients(loss_train)
    # capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gvs]
    # op = optimizer.apply_gradients(capped_gvs)

    op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_train)

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    if os.path.isfile('model.npz'):
        tl.files.load_and_assign_npz(sess, 'model.npz', net_train)

    for it in range(n_epochs):
        tot_loss = tot_loss_1 = tot_loss_2 = tot_loss_3 = tot_loss_4 = 0
        for num in range(n_batches):
            imgs, labels = read_data(num, n_batches)
            feed_dict = {X: imgs, y_: labels}
            _, loss, loss_1, loss_2, loss_3, loss_4, pred = sess.run([op, loss_train, coord_loss, len_loss, is_obj_loss, no_obj_loss, y_train], feed_dict=feed_dict)
            tot_loss += loss
            tot_loss_1 += loss_1
            tot_loss_2 += loss_2
            tot_loss_3 += loss_3
            tot_loss_4 += loss_4
        if it % 10 == 0:
            print(pred[0, :, :, 0])
            show_bbox(imgs[0], pred[0])
            show_bbox(imgs[0], labels[0])
        if it % 10 == 0:
            tl.files.save_npz(net_train.all_params, 'model' + str(it), sess)
        print('epoch %d\nloss: %f\ncoord loss: %f\nlen loss: %f\nis object loss: %f\nno object loss: %f' %
              (it, tot_loss / n_batches, tot_loss_1 / n_batches, tot_loss_2 / n_batches, tot_loss_3 / n_batches, tot_loss_4 / n_batches))

    tl.files.save_npz(net_train.all_params, 'model.npz', sess)


if __name__ == '__main__':
    main()
