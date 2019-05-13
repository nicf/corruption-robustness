# This a modified version of code available on GitHub at
# https://github.com/MadryLab/cifar10_challenge, where it appears with the
# following license:
#
# MIT License
#
# Copyright (c) 2017 Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt,
# Dimitris Tsipras, and Adrian Vladu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""
Utilities for importing the CIFAR10 dataset.

Each image in the dataset is a numpy array of shape (32, 32, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys

import numpy as np

version = sys.version_info


class CIFAR10Data(object):
  """
  Unpickles the CIFAR10 dataset from a specified folder containing a pickled
  version following the format of Krizhevsky which can be found
  [here](https://www.cs.toronto.edu/~kriz/cifar.html).

  Inputs to constructor
  =====================

      - path: path to the pickled dataset. The training data must be pickled
      into five files named data_batch_i for i = 1, ..., 5, containing 10,000
      examples each, the test data
      must be pickled into a single file called test_batch containing 10,000
      examples, and the 10 class names must be
      pickled into a file called batches.meta. The pickled examples should
      be stored as a tuple of two objects: an array of 10,000 32x32x3-shaped
      arrays, and an array of their 10,000 true labels.

  """

  def __init__(self, path):
    train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
    eval_filename = 'test_batch'
    metadata_filename = 'batches.meta'

    train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
    train_labels = np.zeros(50000, dtype='int32')
    for ii, fname in enumerate(train_filenames):
      cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
      train_images[ii * 10000 : (ii+1) * 10000, Ellipsis] = cur_images
      train_labels[ii * 10000 : (ii+1) * 10000, Ellipsis] = cur_labels
    eval_images, eval_labels = self._load_datafile(
        os.path.join(path, eval_filename))

    self.train_data = DataSubset(train_images, train_labels)
    self.eval_data = DataSubset(eval_images, eval_labels)

  @staticmethod
  def _load_datafile(filename):
    with open(filename, 'rb') as fo:
      if version.major == 3:
        data_dict = pickle.load(fo, encoding='bytes')
      else:
        data_dict = pickle.load(fo)

      assert data_dict[b'data'].dtype == np.uint8
      image_data = data_dict[b'data']
      image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
      return image_data, np.array(data_dict[b'labels'])


class DataSubset(object):

  def __init__(self, xs, ys):
    self.xs = xs.astype(np.float) / 255.0
    self.n = xs.shape[0]
    self.ys = ys
    self.batch_start = 0
    self.cur_order = np.random.permutation(self.n)

  def get_next_batch(self, batch_size, multiple_passes=False,
                     reshuffle_after_pass=True):
    if self.n < batch_size:
      raise ValueError('Batch size can be at most the dataset size')
    if not multiple_passes:
      actual_batch_size = min(batch_size, self.n - self.batch_start)
      if actual_batch_size <= 0:
        raise ValueError('Pass through the dataset is complete.')
      batch_end = self.batch_start + actual_batch_size
      batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], Ellipsis]
      batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], Ellipsis]
      self.batch_start += actual_batch_size
      return batch_xs, batch_ys
    actual_batch_size = min(batch_size, self.n - self.batch_start)
    if actual_batch_size < batch_size:
      if reshuffle_after_pass:
        self.cur_order = np.random.permutation(self.n)
      self.batch_start = 0
    batch_end = self.batch_start + batch_size
    batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], Ellipsis]
    batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], Ellipsis]
    self.batch_start += batch_size

    return batch_xs, batch_ys
