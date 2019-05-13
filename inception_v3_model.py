"""Code for loading an Inception v3 model for ImageNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from adv_corr_robust import model
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception

IMAGE_SIZE = 299
NUM_CLASSES = 1001


class InceptionV3Model(model.Model):
  """The Inceptionv3 model."""

  def __init__(self):
    super(InceptionV3Model, self).__init__(
        features_shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3],
        num_classes=NUM_CLASSES)

  def build_logits(self):
    features = self.features

    with slim.arg_scope(inception.inception_v3_arg_scope()):
      self.logits, _ = inception.inception_v3(
          features, num_classes=NUM_CLASSES, is_training=False, reuse=None)

  def get_variables_to_restore(self):
    return slim.get_variables_to_restore()

  def load(self, session, filename):
    self.saver.restore(session, filename)
