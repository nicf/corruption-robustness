"""Some helper functions for generating the plots."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def flatten(x):
  shape = x.shape
  flat_shape = np.prod(shape[1:])
  return np.reshape(x, [-1, flat_shape])


def get_dists(x1, x2):
  assert len(x1.shape) > 1 and len(x2.shape) > 1, 'bad shapes: {}, {}'.format(
      x1.shape, x2.shape)
  x1 = flatten(x1)
  x2 = flatten(x2)

  diff = x1 - x2
  return [np.linalg.norm(d) for d in diff]


def distances_to_nearest_error(
    model, sess, xs, ys, sigma, adjust_step_size, distance_scale,
    sample_batch_size=10):
  """The distances from each provided point to its nearest error.

  This function runs a PGD attack on each point provided and reports the l2
  distance to the nearest error the attack finds for each one, together with
  the number of attacks that were successful.

  Args:
    model: A `Model`; the model to attack.
    sess: A `tf.Session` with the relevant model parameters loaded.
    xs: An `np.array` containing the images to attack.
    ys: An `np.array` containing the correct labels for each image.
    sigma: The scale of Gaussian noise being evaluated. This is used to pick
      some attack parameters intelligently.
    adjust_step_size: If true, use a smaller step size.
    distance_scale: The difference between the minimum and maximum values for
      each pixel. Used to inform the choice of attack parameters.
    sample_batch_size: The number of images to attack at once.

  Returns:
    A tuple containing an `np.array` of distances and an `int`, the number of
      successful attacks.
  """

  n = np.random.normal(scale=sigma, size=1000000)
  y = np.percentile(n, 100 * (1 - 1e-5))
  epsilon = max(y * 2.0, 1.0 * distance_scale)
  step_size = epsilon / 150.0

  if adjust_step_size:
    step_size /= 1.5

  tf.logging.info('using epsilon={}, step_size={}'.format(epsilon, step_size))

  start = 0
  success = 0
  x_adv = []
  while start < xs.shape[0]:
    batch_x_adv, batch_success = model.pgd_attack(
        xs[start:start + sample_batch_size],
        ys[start:start + sample_batch_size],
        epsilon, sess, step_size=step_size,
        random_start=False, steps=200, norm='l2', stop_early=True,
        clip_to=None, verbose=False, return_success=True)
    x_adv.extend(list(batch_x_adv))
    success += batch_success
    start += sample_batch_size

  x_adv = np.array(x_adv)
  return get_dists(x_adv, xs), success


def error_rate_in_noise_with_good_examples(
    model, sess, x, y, sigma, num_examples_wanted, gaussian_samples_at_sigma_1,
    sample_batch_size=10):
  """The error rate in Gaussian noise together with some good examples.

  Returns the error rate in Gaussian noise around the provided image together
  with an array of correctly classified noisy images at the provided noise
  scale. Some samples from a Gaussian at sigma=1 also must be provided, since
  this function is designed to use the sam

  Args:
    model: A `Model`; the model to attack.
    sess: A `tf.Session` with the relevant model parameters loaded.
    x: An `np.array` containing the image to attack.
    y: An `np.array` containing the correct label for each image.
    sigma: The scale of Gaussian noise being evaluated.
    num_examples_wanted: The number of good example to return.
    gaussian_samples_at_sigma_1: An `np.array` containing samples from a
      Gaussian at sigma=1.
    sample_batch_size: The number of images to attack at once.


  Returns:
    A tuple containing a float and an `np.array`, which are the error rate in
      noise and a batch of correctly classified examples.
  """
  noisy_x = x + sigma * gaussian_samples_at_sigma_1
  total_iters = noisy_x.shape[0]

  correct_examples = []
  start = 0
  num_correct = 0
  while start < total_iters:
    if gaussian_samples_at_sigma_1 is not None:
      batch_noisy_x = noisy_x[start:start + sample_batch_size]
    else:
      batch_noisy_x = x + np.random.normal(
          scale=sigma,
          size=[sample_batch_size] + list(x.shape)[1:])
    tiled_y = np.tile(y, reps=batch_noisy_x.shape[0])

    batch_correct = sess.run(model.correct, feed_dict={
        model.features: batch_noisy_x,
        model.labels: tiled_y
    })
    if len(correct_examples) < num_examples_wanted:
      correct_examples.extend(list(batch_noisy_x[batch_correct]))
    num_correct += np.sum(batch_correct)
    start += sample_batch_size

  correct_examples = np.array(correct_examples)
  error_rate = 1.0 - float(num_correct) / float(total_iters)
  num_to_grab = min(num_examples_wanted, correct_examples.shape[0])

  return error_rate, correct_examples[:num_to_grab]


def sigma_at_error_rate_with_good_examples(
    model, sess, x, y, desired_error_rate,
    gaussian_samples_at_sigma_1, num_examples_wanted, distance_scale,
    initial_guess=0.1, tol=0.001, sample_batch_size=10):
  """The scale at which Gaussian noise produces the provided error rate.

  Args:
    model: A `Model`; the model to attack.
    sess: A `tf.Session` with the relevant model parameters loaded.
    x: An `np.array` containing the image to attack.
    y: An `np.array` containing the correct label for each image.
    desired_error_rate: The error rate to look for.
    gaussian_samples_at_sigma_1: An `np.array` containing samples from a
      Gaussian at sigma=1.
    num_examples_wanted: The number of good example to return.
    distance_scale: The difference between the minimum and maximum values for
      each pixel. Used to inform the initialization of the search.
    initial_guess: Where to start the binary search.
    tol: The tolerance of the search; we stop when the bounds are at most this
      close.
    sample_batch_size: The number of images to attack at once.


  Returns:
    A tuple containing two floats and an `np.array`, which are the sigma at the
      conclusion of the search, the error rate at that sigma, and a batch of
      correctly classified examples.
  """
  initial_guess *= distance_scale
  tol *= distance_scale

  lower_bound = 0.0
  upper_bound = 1.0 * distance_scale
  current_guess = initial_guess

  while upper_bound - lower_bound > tol:
    error_rate, good_examples = error_rate_in_noise_with_good_examples(
        model, sess, x, y, sigma=current_guess,
        gaussian_samples_at_sigma_1=gaussian_samples_at_sigma_1,
        num_examples_wanted=num_examples_wanted,
        sample_batch_size=sample_batch_size)

    if error_rate > desired_error_rate:
      old_guess = current_guess
      current_guess = (lower_bound + current_guess) / 2.0
      upper_bound = old_guess
    else:
      old_guess = current_guess
      current_guess = min(current_guess * 2.0,
                          (current_guess + upper_bound) / 2.0)
      lower_bound = old_guess

  return current_guess, error_rate, good_examples


def expand_k_times(c, k):
  assert len(c.shape) == 1
  for _ in xrange(k):
    c = np.expand_dims(c, axis=1)
  return c
