import os.path
import shutil
import time
from glob import glob
import random
import numpy as np
import scipy.misc
import tensorflow as tf
from keras.utils import get_file
from tqdm import tqdm

class_car = 1
class_road = 2
n_classes = 3


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_mobilenet(data_dir):
    alpha_text = '1_0'
    rows = 224
    mobilenet_path = os.path.join(data_dir, 'mobilenet')
    model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
    mobilenet_file = os.path.join(mobilenet_path, model_name)

    if not os.path.exists(mobilenet_file):
        if os.path.exists(mobilenet_path):
            shutil.rmtree(mobilenet_path)
        os.makedirs(mobilenet_path)

    BASE_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'  # noqa
    weigh_path = BASE_WEIGHT_PATH + model_name
    weight_path = get_file(
        os.path.abspath(mobilenet_file), weigh_path, cache_subdir='models')
    return weight_path


def maybe_download_pretrained_mobilenetv2(data_dir):
    alpha = '1.0'
    rows = 224
    mobilenet_path = os.path.join(data_dir, 'mobilenet')
    model_name = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + str(alpha) + '_' + str(rows) + '_no_top' + '.h5'
    mobilenet_file = os.path.join(mobilenet_path, model_name)

    if not os.path.exists(mobilenet_file):
        if os.path.exists(mobilenet_path):
            shutil.rmtree(mobilenet_path)
        os.makedirs(mobilenet_path)

    BASE_WEIGHT_PATH = 'https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/'  # noqa
    weight_path = BASE_WEIGHT_PATH + model_name
    weights_path = get_file(os.path.abspath(mobilenet_file), weight_path,
                            cache_subdir='models')
    return weights_path


def flip_img(image, gt_bg, gt_car, gt_road):
    image = np.fliplr(image)
    gt_bg = np.fliplr(gt_bg)
    gt_car = np.fliplr(gt_car)
    gt_road = np.fliplr(gt_road)
    return image, gt_bg, gt_car, gt_road


def gen_batches_functions(data_folder, image_paths, image_shape, out_shape,
                          label_folder):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        id_road = 7
        id_lane = 6
        id_car = 10

        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                # Get corresponding label img path
                gt_image_file = image_file.replace('CameraRGB', 'CameraSeg')
                # Read rgb and label images
                img_in = scipy.misc.imread(image_file, mode='RGB')
                gt_in = scipy.misc.imread(gt_image_file)
                # Crop sky part of the image
                image = img_in[-out_shape[0]:, :]
                gt_image = gt_in[-out_shape[0]:, :, 0]
                # Obtain labels
                gt_road = ((gt_image == id_road) | (gt_image == id_lane))
                gt_car = (gt_image == id_car)
                gt_car[-105:, :] = False
                gt_bg = np.invert(gt_car | gt_road)
                # Augmentation
                if bool(random.getrandbits(1)):
                    image, gt_bg, gt_car, gt_road = flip_img(
                        image, gt_bg, gt_car, gt_road)

                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_car = gt_car.reshape(*gt_car.shape, 1)
                gt_road = gt_road.reshape(*gt_road.shape, 1)

                gt_image = np.concatenate((gt_bg, gt_car, gt_road), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn


# def blend_output(frame, im_out, image_shape):

#     car_segmentation = np.where((im_out == class_car), 1,
#                                 0).astype('uint8').reshape(
#                                     image_shape[0], image_shape[1], 1)
#     car_mask = np.dot(car_segmentation, np.array([[0, 255, 0, 127]]))
#     car_mask = scipy.misc.toimage(car_mask, mode="RGBA")

#     road_segmentation = np.where((im_out == class_road), 1,
#                                  0).astype('uint8').reshape(
#                                      image_shape[0], image_shape[1], 1)
#     road_mask = np.dot(road_segmentation, np.array([[255, 0, 0, 127]]))
#     road_mask = scipy.misc.toimage(road_mask, mode="RGBA")

#     street_im = scipy.misc.toimage(frame)
#     street_im.paste(road_mask, box=None, mask=road_mask)
#     street_im.paste(car_mask, box=None, mask=car_mask)

#     return street_im


def get_seg_img(sess, logits, image_pl, pimg_in, image_shape, nw_shape,
                learning_phase):
    im_out = np.zeros(image_shape)

    pimg = pimg_in[-nw_shape[0]:, :, :]
    im_softmax = sess.run(
        tf.nn.softmax(logits), {
            image_pl: [pimg],
            learning_phase: 0
        })

    im_softmax = im_softmax.reshape(nw_shape[0], nw_shape[1], -1)
    im_out[-nw_shape[0]:, :] = im_softmax.argmax(axis=2)

    image = scipy.misc.toimage(pimg_in)

    car_segmentation = np.where((im_out == class_car), 1,
                                0).astype('uint8').reshape(
                                    image_shape[0], image_shape[1], 1)
    car_mask = np.dot(car_segmentation, np.array([[0, 255, 0, 127]]))
    car_mask = scipy.misc.toimage(car_mask, mode="RGBA")

    road_segmentation = np.where((im_out == class_road), 1,
                                 0).astype('uint8').reshape(
                                     image_shape[0], image_shape[1], 1)
    road_mask = np.dot(road_segmentation, np.array([[255, 0, 0, 127]]))
    road_mask = scipy.misc.toimage(road_mask, mode="RGBA")

    street_im = scipy.misc.toimage(image)
    street_im.paste(road_mask, box=None, mask=road_mask)
    street_im.paste(car_mask, box=None, mask=car_mask)

    return street_im


def gen_test_output(sess, logits, image_folder, image_pl, data_folder,
                    learning_phase, image_shape, nw_shape):
    """
    Generate test output using the test images
    """
    image_paths = glob(os.path.join(data_folder, image_folder, '*.png'))
    for image_file in image_paths[:5]:

        in_image = scipy.misc.imread(image_file, mode='RGB')
        image = scipy.misc.imresize(in_image, image_shape)

        street_im = get_seg_img(sess, logits, image_pl, image, image_shape,
                                nw_shape, learning_phase)

        street_im = scipy.misc.imresize(street_im, in_image.shape)
        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, image_folder, sess, image_shape,
                           nw_shape, logits, learning_phase, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, image_folder, input_image,
                                    data_dir, learning_phase, image_shape,
                                    nw_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
