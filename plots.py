"""Generates data for the plots in the paper.

This code is meant to accompany the paper "Adversarial Examples Are a Natural
Consequence of Test Error in Noise". It is used to generate the data appearing
in the paper illustrating the relationship between adversarial examples and
errors in the presence of Gaussian noise.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import os

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
from adv_corr_robust import datasets
from adv_corr_robust import inception_v3_model
from adv_corr_robust import madry_cifar_model
from adv_corr_robust import utils

flags.DEFINE_string('model', None, '"cifar", or "imagenet".')

flags.DEFINE_string('model_file', None, 'The model checkpoint file to load.')
flags.DEFINE_string('data_dir', None, 'Directory to load data from.')
flags.DEFINE_string('save_dir', None, 'Directory to save results to.')

flags.DEFINE_integer('batch_size', 64, 'batch size')

flags.DEFINE_integer('num_batches', 10, 'number of batches to evaluate')
flags.DEFINE_integer('num_samples', 50000, 'number of noisy points')
flags.DEFINE_integer('num_pgd_points', 200,
                     'number of noisy points to plug into PGD')

flags.DEFINE_float('sigma', 0.1, 'scale of the noise')
flags.DEFINE_float('desired_error_rate', 0.01, 'desired error rate')

FLAGS = flags.FLAGS


def combined_plot_data(model, dataset_generator, num_batches,
                       noise_samples_per_point, num_points_for_pgd,
                       initial_noise_sigma, distance_scale, desired_error_rate,
                       session):
  """Get the data for the combined plot."""
  clean_dists = []
  clean_pgd_successes = []
  initial_error_rates = []
  sigmas_at_desired_error_rate = []

  initial_distances_to_noisy_lists = []
  initial_noisy_pgd_successes = []

  estimates_of_desired_error_rate = []
  desired_distances_to_noisy_lists = []
  desired_noisy_pgd_successes = []

  done = 0
  for _ in range(num_batches):
    tf.logging.info('running on a batch')
    xs, ys = next(dataset_generator)
    tf.logging.info('xs: {}, ys: {}'.format(xs.dtype, ys.dtype))
    for idx in range(xs.shape[0]):
      x = xs[idx:idx+1]
      y = ys[idx:idx+1]
      tf.logging.info('the label for this image is {}'.format(y))

      correct = session.run(
          model.correct, feed_dict={model.features: x, model.labels: y})[0]
      if not correct:
        tf.logging.info('this point is incorrect; skipping')
        clean_dists.append(0.0)
        clean_pgd_successes.append(1)
        initial_error_rates.append(1.0)
        sigmas_at_desired_error_rate.append(0.0)
        estimates_of_desired_error_rate.append(1.0)
        initial_distances_to_noisy_lists.append([0.0])
        initial_noisy_pgd_successes.append(num_points_for_pgd)
        desired_distances_to_noisy_lists.append([0.0])
        desired_noisy_pgd_successes.append(num_points_for_pgd)
        continue

      tf.logging.info('  getting distance from clean point...')
      [clean_dist], clean_success = utils.distances_to_nearest_error(
          model, session, x, y, sigma=initial_noise_sigma,
          adjust_step_size=False, distance_scale=distance_scale)
      tf.logging.info('  distance = {}'.format(clean_dist))
      tf.logging.info('  success = {}'.format(clean_success))

      tf.logging.info(
          '  getting initial error rate in noise (sigma={})...'.format(
              initial_noise_sigma))
      initial_n = np.random.normal(
          scale=1.0, size=[noise_samples_per_point] + list(x.shape)[1:])
      (initial_error_rate,
       initial_noisy_xs) = utils.error_rate_in_noise_with_good_examples(
           model, session, x, y, sigma=initial_noise_sigma,
           num_examples_wanted=num_points_for_pgd,
           gaussian_samples_at_sigma_1=initial_n)
      tf.logging.info('  error rate = {}'.format(initial_error_rate))

      n = np.random.normal(
          scale=1.0, size=[1000] + list(x.shape)[1:])

      tf.logging.info(
          '  getting sigma at desired error rate (={})...'.format(
              desired_error_rate))
      (sigma_at_desired_error_rate,
       estimate_of_desired_error_rate,
       desired_noisy_xs) = utils.sigma_at_error_rate_with_good_examples(
           model, session, x, y, desired_error_rate=desired_error_rate,
           gaussian_samples_at_sigma_1=n,
           num_examples_wanted=num_points_for_pgd,
           distance_scale=distance_scale)
      tf.logging.info('  sigma = {}'.format(sigma_at_desired_error_rate))

      tiled_ys = np.tile(y, reps=initial_noisy_xs.shape[0])

      tf.logging.info(
          '  getting distance from noisy points at sigma={}'.format(
              initial_noise_sigma))
      if initial_error_rate < 0.5:
        (init_distances_to_noisy,
         init_noisy_success) = utils.distances_to_nearest_error(
             model, session, xs=initial_noisy_xs, ys=tiled_ys,
             sigma=initial_noise_sigma,
             adjust_step_size=initial_error_rate > 0.1,
             distance_scale=distance_scale)
      else:
        init_distances_to_noisy, init_noisy_success = (
            [0.0], initial_noisy_xs.shape[0])
      tf.logging.info('  avg dist = {}'.format(
          np.mean(init_distances_to_noisy)))
      tf.logging.info('  success = {}'.format(init_noisy_success))

      tiled_ys = np.tile(y, reps=desired_noisy_xs.shape[0])

      tf.logging.info(
          '  getting distance from noisy points at sigma={}'.format(
              sigma_at_desired_error_rate))
      (desired_distances_to_noisy,
       desired_noisy_success) = utils.distances_to_nearest_error(
           model, session, xs=desired_noisy_xs, ys=tiled_ys,
           sigma=sigma_at_desired_error_rate,
           adjust_step_size=True,
           distance_scale=distance_scale)
      tf.logging.info('  avg dist = {}'.format(
          np.mean(desired_distances_to_noisy)))
      tf.logging.info('  success = {}'.format(desired_noisy_success))

      # This is the x axis of the plots in Figures 2, and the
      #  plots on the bottom of Figure 4.
      sigmas_at_desired_error_rate.append(sigma_at_desired_error_rate)

      # This is the x axis of the plots on the top of Figure 4.
      initial_error_rates.append(initial_error_rate)

      # This is the y axis of the plots in Figure 2.
      clean_dists.append(clean_dist)

      # This is the y axis of the plots in Figure 4.
      initial_distances_to_noisy_lists.append(init_distances_to_noisy)

      initial_noisy_pgd_successes.append(init_noisy_success)
      estimates_of_desired_error_rate.append(estimate_of_desired_error_rate)
      desired_distances_to_noisy_lists.append(desired_distances_to_noisy)
      desired_noisy_pgd_successes.append(desired_noisy_success)

      clean_pgd_successes.append(clean_success)

      tf.logging.info('Finished {} points'.format(done))
      done += 1

  return {
      'clean_dists': clean_dists,
      'clean_pgd_successes': clean_pgd_successes,
      'initial_error_rates': initial_error_rates,
      'sigmas_at_desired_error_rate': sigmas_at_desired_error_rate,
      'estimates_of_desired_error_rate': estimates_of_desired_error_rate,
      'initial_distances_to_noisy_lists': initial_distances_to_noisy_lists,
      'initial_noisy_pgd_successes': initial_noisy_pgd_successes,
      'desired_distances_to_noisy_lists': desired_distances_to_noisy_lists,
      'desired_noisy_pgd_successes': desired_noisy_pgd_successes,
      'sigma': initial_noise_sigma,
      'desired_error_rate': desired_error_rate,
      'distance_scale': distance_scale
  }


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  assert FLAGS.model is not None
  assert FLAGS.model_file is not None
  assert FLAGS.save_dir is not None
  assert FLAGS.data_dir is not None

  if FLAGS.model == 'cifar':
    model_class = madry_cifar_model.MadryCifarModel
    data_gen = datasets.cifar_gen(
        FLAGS.data_dir, FLAGS.batch_size)
    sigma_rescale = 1.0
  elif FLAGS.model == 'imagenet':
    model_class = inception_v3_model.InceptionV3Model
    data_gen = datasets.inception_imagenet_gen(
        FLAGS.data_dir, FLAGS.batch_size)
    sigma_rescale = 2.0
  else:
    raise ValueError('Unknown model class {}'.format(FLAGS.model))

  tf.logging.info('Loading model {} from {}'.format(
      FLAGS.model, FLAGS.model_file))

  model = model_class()
  sess = model.new_session()
  model.load(sess, FLAGS.model_file)

  tf.logging.info('Model loaded')

  tf.logging.info(
      'num_batches={}, noise_samples_per_point={}, num_points_for_pgd={}, '
      'initial_noise_sigma={}, desired_error_rate={}'.format(
          FLAGS.num_batches, FLAGS.num_samples, FLAGS.num_pgd_points,
          FLAGS.sigma*sigma_rescale, FLAGS.desired_error_rate))

  data = combined_plot_data(
      model=model,
      dataset_generator=data_gen,
      num_batches=FLAGS.num_batches,
      noise_samples_per_point=FLAGS.num_samples,
      num_points_for_pgd=FLAGS.num_pgd_points,
      initial_noise_sigma=FLAGS.sigma * sigma_rescale,
      desired_error_rate=FLAGS.desired_error_rate,
      session=sess,
      distance_scale=sigma_rescale)

  out_path = os.path.join(FLAGS.save_dir, 'combined_plot.pkl')
  with tf.gfile.Open(out_path, 'w') as f:
    pickle.dump(data, f)

if __name__ == '__main__':
  app.run(main)
