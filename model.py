"""A class representing a trained model for the purpose of evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import os
import numpy as np
import tensorflow as tf

from adv_corr_robust import utils


def get_norms(x):
  tmp = utils.flatten(x)
  return [np.linalg.norm(tmp[i]) for i in xrange(len(tmp))]


def l2_clip(cur, x, epsilon):
  new_shape = (-1,) + (1,) * (len(cur.shape) - 1)

  norms = np.reshape(np.array(get_norms(cur - x)), new_shape)
  condition = np.reshape(norms > epsilon, new_shape)

  return np.where(
      condition,
      x + epsilon * (cur - x) / norms,
      cur
  )


def _project_perturbation(perturbation, epsilon, input_image):
  """Project `perturbation` onto L-infinity ball of radius `epsilon`."""
  clipped_perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
  new_image = tf.clip_by_value(input_image + clipped_perturbation, 0., 1.)
  return new_image - input_image


class Model(object):
  """A model on which to do the adversarial things."""

  def __init__(self, features_shape, num_classes=10, scope=''):
    self._g = tf.Graph()

    with self._g.as_default():
      with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        self.build(features_shape, num_classes)
      self._vars = tf.trainable_variables()

      self.saver = tf.train.Saver(self.get_variables_to_restore())

  def get_variables_to_restore(self):
    return None

  def new_session(self, master=None):
    """Create a new session for use with this model."""
    with self._g.as_default():
      if master is not None:
        return tf.Session(master)
      else:
        return tf.Session()

  def save(self, session, filename):
    self.saver.save(session, filename)

  def load(self, session, filename):
    self.saver.restore(session, filename)

  def build_logits(self):
    raise NotImplementedError()

  def preprocess_data(self, raw_features):
    """Return a processed version of  `raw_features`."""
    return raw_features

  def build(self, features_shape, num_classes):
    """Build the graph for the model."""

    self.features = tf.placeholder(shape=features_shape, dtype=tf.float32)
    self.labels = tf.placeholder(shape=[None], dtype=tf.int32)

    self.build_logits()

    self.probs = tf.nn.softmax(self.logits)
    self.pred = tf.argmax(self.probs, 1)

    self.log_probs = tf.nn.log_softmax(self.logits)
    self.mean_log_probs = tf.reduce_mean(self.log_probs, axis=0)

    self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.logits, labels=self.labels)

    self.loss = tf.reduce_mean(self.losses)
    [self.loss_grad] = tf.gradients(self.loss, self.features)
    self.correct = tf.equal(
        tf.cast(tf.argmax(self.logits, 1), self.labels.dtype),
        self.labels)
    self.accuracy = tf.reduce_mean(tf.to_float(self.correct))

    self.num_classes = num_classes

  def pgd_attack(self, x, y, epsilon, session, steps=40, step_size=0.01,
                 norm='linf', random_start=False, verbose=False, attempts=1,
                 loss='xent', target_class=None, stop_early=False,
                 clip_to=1.0, clip_min=0.0, return_success=False, **kwargs):
    """The result of a PGD attack on the model."""
    if verbose:
      print('running attack with epsilon={}, steps={}, step_size={},'
            ' norm={}, attempts={}, random_start={}'.format(
                epsilon, steps, step_size, norm, attempts, random_start))

    if loss == 'xent':
      if target_class is None:
        loss_tensor = self.loss
        loss_grad_tensor = self.loss_grad
        labels_for_feed_dict = y
      else:
        loss_tensor = -self.loss
        loss_grad_tensor = -self.loss_grad
        if len(np.shape(target_class)) == 1:
          labels_for_feed_dict = np.array(target_class)
        else:
          labels_for_feed_dict = target_class * np.ones_like(y)
    elif loss == 'cw':
      loss_tensor = self.cw_loss
      loss_grad_tensor = self.cw_loss_grad
      labels_for_feed_dict = y
    else:
      raise ValueError('Unrecognized loss: {}'.format(loss))

    if steps >= 100:
      report_interval = steps // 25
    else:
      report_interval = 1

    indices_this_attempt = list(range(x.shape[0]))
    cur = np.zeros_like(x)
    for attempt in range(attempts):
      if verbose:
        print('attempt #{}; trying {} examples'.format(
            attempt, len(indices_this_attempt)))

      indices = indices_this_attempt[:]  # copy the list

      if random_start:
        if norm == 'linf':
          cur[indices] = x[indices] + np.random.uniform(
              -epsilon, epsilon, x[indices].shape)
        elif norm == 'l2':
          size = 1.0
          for dim in x.shape[1:]:
            size *= float(dim)
          cur[indices] = x[indices] + np.random.normal(
              scale=epsilon/np.sqrt(size), size=x[indices].shape)
        else:
          raise ValueError('norm should be "linf" or "l2"')
      else:
        cur[indices] = np.copy(x[indices])

      for i in range(steps):
        correct, pred = session.run(
            [self.correct, self.pred],
            feed_dict={self.features: cur[indices],
                       self.labels: labels_for_feed_dict[indices]})

        if stop_early:
          if target_class is not None:
            should_not_stop = (pred != labels_for_feed_dict[indices])
          else:
            should_not_stop = correct
          # Keep only the indices that are still classified right.
          new_indices = []
          for idx in range(len(indices)):
            if should_not_stop[idx]:
              new_indices.append(indices[idx])
          indices = new_indices

          # No need to continue if we've found something everywhere.
          if not indices:
            break

        gradient, loss = session.run(
            [loss_grad_tensor, loss_tensor],
            feed_dict={self.features: cur[indices],
                       self.labels: labels_for_feed_dict[indices]})

        if norm == 'linf':
          grad_norm = utils.expand_k_times(
              np.array(get_norms(gradient)), len(x.shape) -1)
          cur[indices] += step_size * np.sign(gradient)
          cur[indices] = np.clip(cur[indices],
                                 x[indices] - epsilon, x[indices] + epsilon)
        elif norm == 'l2':
          grad_norm = utils.expand_k_times(
              np.array(get_norms(gradient)), len(x.shape) -1)
          cur[indices] = np.where(
              grad_norm > 1e-15,
              cur[indices] + step_size * (gradient / grad_norm),
              cur[indices]
          )
          cur[indices] = l2_clip(cur[indices], x[indices], epsilon)
        else:
          raise ValueError('norm should be "linf" or "l2"')

        if clip_to is not None:
          cur[indices] = np.clip(cur[indices], clip_min, clip_to)

        if verbose and i % report_interval == 0:
          l2 = []
          linf = []
          for j in range(cur.shape[0]):
            noise = cur[j] - x[j]
            noise = np.reshape(noise, [-1])

            l2.append(np.linalg.norm(noise))
            linf.append(np.linalg.norm(noise, ord=np.inf))

          correct = session.run(
              self.correct,
              feed_dict={self.features: cur,
                         self.labels: labels_for_feed_dict})
          acc = np.mean(correct)
          print('step {}/{}\texamples: {}\tloss: {}\tl^2: {}\tl^inf: {}'
                '\tacc: {}'.format(
                    i, steps, len(indices),
                    loss, np.mean(l2), np.mean(linf), acc))

      # Keep the indices that were still classified right for the next attempt.
      new_indices = []
      for idx in range(len(indices)):
        if correct[idx]:
          new_indices.append(indices[idx])
      indices_this_attempt = new_indices

      if not indices_this_attempt:
        break

    if return_success:
      return cur, x.shape[0] - len(indices_this_attempt)
    else:
      return cur
