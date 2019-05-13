"""Code for loading the CIFAR and ImageNet datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import tensorflow as tf
from adv_corr_robust import cifar10_input


def inception_imagenet_datasets(dataset_dir):
  """Returns ImageNet data for use in the Model class."""
  image_size = 299

  def _get_split(split_name):
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
      split_name: A train/test split name.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/validation split.
    """
    if split_name not in ['train', 'validation']:
      raise ValueError('split name %s was not recognized.' % split_name)

    file_pattern = os.path.join(dataset_dir, split_name + '*')

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature(
            [1], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(
            dtype=tf.int64),
    }

    def _decode_image(image, height, width,
                      central_fraction=0.875, scope=None):
      """Decode an image from the TFRecord file."""
      with tf.name_scope(scope, 'eval_image', [image, height, width]):
        image = tf.image.decode_jpeg(image, channels=3)
        if image.dtype != tf.float32:
          image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_fraction:
          image = tf.image.central_crop(
              image, central_fraction=central_fraction)

        if height and width:
          # Resize the image to the specified height and width.
          image = tf.expand_dims(image, 0)
          image = tf.image.resize_bilinear(image, [height, width],
                                           align_corners=False)
          image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

    def _parse_function(example_proto):
      parsed_features = tf.parse_single_example(
          example_proto, keys_to_features)
      return {
          'features': _decode_image(
              parsed_features['image/encoded'],
              height=image_size, width=image_size),
          'labels': tf.squeeze(parsed_features['image/class/label'])
      }

    dataset = tf.data.TFRecordDataset(tf.gfile.Glob(file_pattern))
    dataset = dataset.map(_parse_function)

    return dataset

  return {
      'test': _get_split('validation')
  }


def cifar_gen(path, batch_size):
  data = cifar10_input.CIFAR10Data(path)
  while True:
    yield data.eval_data.get_next_batch(batch_size)


def inception_imagenet_gen(path, batch_size):
  datasets = inception_imagenet_datasets(path)
  test_dataset = datasets['test'].batch(batch_size)
  iterator = test_dataset.make_initializable_iterator()
  features_and_labels = iterator.get_next()

  with tf.Session() as data_session:
    data_session.run(iterator.initializer)
    while True:
      try:
        d = data_session.run(features_and_labels)
        yield d['features'], d['labels']
      except tf.errors.OutOfRangeError:
        break
