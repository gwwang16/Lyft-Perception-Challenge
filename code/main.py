import os
import os.path
import shutil
import warnings
from glob import glob

import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import helper2

from seg_mobilenetv1 import SegMobileNet_u

class_bg = 0
class_car = 1
class_road = 2
n_classes = 3

image_shape = (600, 800)
out_shape = (416, 800)
# make out_shape divisible
out_shape = [x // 32 * 32 for x in out_shape]

KERAS_TRAIN = 1
KERAS_TEST = 0
LEARNING_RATE = 1e-4  # 1e-3

# Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn(
#         'No GPU found. Please use a GPU to train your neural network.')
# else:
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def F_beta(beta, precision, recall):
    """Weighted F beta score"""
    epsilon = 1e-5
    f_beta = (1 + beta**2) * (precision * recall) / ((beta**2) * precision + recall + epsilon)
    return f_beta


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # Bg, car, road
    weights = [0.2, 10, 0.3]  

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    #     labels=correct_label, logits=logits)
    # cross_entropy_loss = tf.reduce_mean(cross_entropy)
    # regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # loss = tf.add(cross_entropy_loss, sum(regularization_losses))
    
    # Classes are unbalanced, add weights to the each class.
    logits_unstack = tf.unstack(logits, axis=1)
    correct_label_unstack = tf.unstack(correct_label, axis=1)
    beta_bg = 1.0
    beta_car = 2.0
    beta_road = 0.5
    for i in range(num_classes):
        logit, label = logits_unstack[i], correct_label_unstack[i]
        num = tf.reduce_sum(tf.multiply(logit, label))
        precision = tf.divide(num, tf.reduce_sum(logit))
        recall = tf.divide(num, tf.reduce_sum(label))
        if i == class_bg:
            f_bg = F_beta(beta_bg, precision, recall)
        if i == class_car:
            f_car = F_beta(beta_car, precision, recall)
        if i == class_road:
            f_road = F_beta(beta_road, precision, recall)
    f_avg = (f_car + f_road) * 0.5

    loss = 10.5 - (weights[0] * f_bg + weights[1] * f_car + weights[2] * f_road)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    return logits, train_op, loss, f_avg


def train_nn(sess, epochs, batch_size, train_batches_fn, val_batches_fn,
             train_op, cross_entropy_loss, input_image, correct_label,
             learning_rate, learning_phase, update_ops, model, f_avg):
    """
    Train neural network and print out the loss during training.
    """

    all_variables = tf.global_variables()
    optimizer_variables = [
        var for var in all_variables
        if var not in model.updates and var not in model.trainable_weights
    ]
    sess.run(tf.variables_initializer(optimizer_variables))
    for var in all_variables:
        sess.run(var)

    # sess.run(tf.global_variables_initializer())
    epoch_pbar = tqdm(range(epochs), ncols=50)
    for epoch in epoch_pbar:
        # train
        train_losses = []
        train_fs = []
        for image, label in train_batches_fn(batch_size):
            fetches = [cross_entropy_loss, f_avg, train_op] + update_ops
            feed_dict = {
                input_image: image,
                correct_label: label,
                learning_rate: LEARNING_RATE,
                learning_phase: KERAS_TRAIN
            }
            train_loss, train_f, *_ = sess.run(fetches, feed_dict=feed_dict)
            train_losses.append(train_loss)
            train_fs.append(train_f)

        # val
        val_losses = []
        val_fs = []
        for image, label in val_batches_fn(batch_size):
            feed_dict = {
                input_image: image[:, -out_shape[0]:, :],
                correct_label: label[:, -out_shape[0]:, :],
                learning_phase: KERAS_TEST
            }
            val_loss, val_f = sess.run([cross_entropy_loss, f_avg], feed_dict=feed_dict)
            val_losses.append(val_loss)
            val_fs.append(val_f)

        epoch_pbar.write("epoch %03d: train_loss: %.4f val_loss: %.4f train_f: %.4f val_f: %.4f" %
                         (epoch, np.mean(train_losses), np.mean(val_losses), np.mean(train_fs), np.mean(val_fs)))

        # save
        if epoch % 1 == 0:
            weight_path = '../model/epoch-%03d-val_loss-%.4f.hdf5' % (
                epoch, np.mean(val_losses))
            model.save(weight_path)


def run():
    load_pretrained = True
    training_state = True

    epochs = 20

    batch_size = 8
    data_dir = '../data/Train_0001/'
    data_dir2 = '../data/Train_0002/'
    data_dir3 = '../data/Train_0003/'
    data_dir4 = '../data/Train_0004/'
    data_dir5 = '../data/Train_0005/'
    test_dir = '../data/Train_0002/'
    runs_dir = '../runs'
    model_dir = '../model'

    if load_pretrained:
        weights_path = helper2.maybe_download_pretrained_mobilenet(model_dir)
        # weights_path = '../model/epoch-009-val_loss-0.1904.hdf5'

    model_files = glob('../model/epoch-*.hdf5')
    model_files.sort(key=os.path.getmtime)

    # training data paths
    image_paths = []
    image_paths2 = []
    image_paths3 = []
    image_paths = glob(os.path.join(data_dir, 'CameraRGB', '*.png'))
    image_paths2 = glob(os.path.join(data_dir2, 'CameraRGB', '*.png'))
    image_paths3 = glob(os.path.join(data_dir3, 'CameraRGB', '*.png'))
    image_paths4 = glob(os.path.join(data_dir4, 'CameraRGB', '*.png'))
    image_paths5 = glob(os.path.join(data_dir5, 'CameraRGB', '*.png'))
    all_image_paths = np.concatenate((image_paths, image_paths2, image_paths3, image_paths4, image_paths5))
    train_paths, val_paths = train_test_split(all_image_paths, test_size=0.1, random_state=23)

    with K.get_session() as sess:
        # Create train and val gen batches
        train_batches_fn = helper2.gen_batches_functions(
            data_dir,
            train_paths,
            image_shape,
            out_shape,
            label_folder='CameraSeg')

        val_batches_fn = helper2.gen_batches_functions(
            data_dir,
            val_paths,
            image_shape,
            out_shape,
            label_folder='CameraSeg')

        learning_phase = K.learning_phase()
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        correct_label = tf.placeholder(
            tf.float32,
            shape=[None, out_shape[0], out_shape[1], n_classes],
            name='correct_label')

        model = SegMobileNet_u(out_shape[0], out_shape[1], num_classes=n_classes)

        # initialize keras variables
        sess = K.get_session()
        if load_pretrained:
            model.load_weights(weights_path, by_name=True)
        input_image = model.input
        logits = model.output
        update_ops = model.updates

        logits, train_op, cross_entropy, f_avg = optimize(
            logits, correct_label, learning_rate, n_classes)

        if training_state:
            print("train sample: %d\tval sample:%d\ttrain steps: %d\tval steps: %d" %
                  (len(train_paths), len(val_paths), int(len(train_paths) / batch_size),
                   int(len(val_paths) / batch_size)))
            train_nn(sess, epochs, batch_size, train_batches_fn,
                     val_batches_fn, train_op, cross_entropy, input_image,
                     correct_label, learning_rate,
                     learning_phase, update_ops, model, f_avg)

        else:
            model.load_weights(model_files[-1])

        helper2.save_inference_samples(
            runs_dir, test_dir, 'CameraRGB', sess, image_shape, out_shape,
            logits, learning_phase, input_image)


if __name__ == '__main__':
    run()
