"""My changed full source code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import keras
import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v2

from PIL import Image
from cleverhans.model import Model

from cleverhans.attacks import FastGradientMethod ,DeepFool
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import tag_constants


slim = tf.contrib.slim



tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', 'inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', './Data/images1/', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', './Data/adv_images/', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.
  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    print(filepath)
   # --- change open method . This change has no impact for code ---
    with open(filepath,"rb") as f:
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.
  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')

# ---change Model from object---
class InceptionModel(Model):
  """Model class for CleverHans library."""

  def __init__(self, nb_classes):
    super(InceptionModel, self).__init__(nb_classes=nb_classes,
                                         needs_dummy_fprop=True)
    self.built = False

  def __call__(self, x_input, return_logits=False):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None

    with tf.Graph().as_default() as g1:
      with tf.Session(graph=g1) as sess:
        input_graph_def = saved_model_utils.get_meta_graph_def(
          "./modelComdef", tag_constants.SERVING).graph_def

        tf.saved_model.loader.load(sess, ["serve"], "./modelComdef")

        g1def = graph_util.convert_variables_to_constants(
          sess,
          input_graph_def,
          ["DY"],
          variable_names_whitelist=None,
          variable_names_blacklist=None)

    with tf.Graph().as_default() as g2:
      with tf.Session(graph=g2) as sess:
        input_graph_def = saved_model_utils.get_meta_graph_def(
          "./newmodel/0", tag_constants.SERVING).graph_def

        tf.saved_model.loader.load(sess, ["serve"], "./newmodel/0")

        g2def = graph_util.convert_variables_to_constants(
          sess,
          input_graph_def,
          ["Tnet", "Tlogits"],
          variable_names_whitelist=None,
          variable_names_blacklist=None)

    with tf.Session() as sess:
      # self.sess1 = sess
      # print("")
      x1 = tf.placeholder("float", [None, 28, 28, 1], name="X")

      y, = tf.import_graph_def(g1def, input_map={"DX:0": x1}, return_elements=["DY:0"])
      net1, logits1 = tf.import_graph_def(g2def, input_map={"TX:0": y}, return_elements=["Tnet:0", "Tlogits:0"])
      self.built = True
      self.logits = logits1
      # Strip off the extra reshape op at the output
      self.probs = net1
    if return_logits:
      return self.logits
    else:
      return self.probs

  def get_logits(self, x_input):
    return self(x_input, return_logits=True)

  def get_probs(self, x_input):
    return self(x_input)

def getFashionmist():
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  x_train = x_train[..., tf.newaxis]
  x_test = x_test[..., tf.newaxis]

  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255

  y_train1 = keras.utils.to_categorical(y_train, 10)
  y_test1 = keras.utils.to_categorical(y_test, 10)
  return x_train, y_train1, y_train, x_test, y_test1, y_test

def main1():
  """Run the sample attack"""
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [None, 28, 28, 1]
  nb_classes = 10

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(nb_classes)

    fgsm = FastGradientMethod(model)
    x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)
    x_train, y_train1, y_train, x_test, y_test1, y_test=getFashionmist()
    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      images=x_test[0:20]
      yt=y_test[0:20]
      adv_images = sess.run(x_adv, feed_dict={x_input: images})
      net=model(adv_images)
      netarg=np.argmax(net)
      pred=model(images)
      prearg=np.argmax(pred)
      for i in range(0,20):

        print("truelabel ",yt[i]," prelabel ",prearg[i]," advlabel ",netarg[i])


if __name__ == '__main__':
  main1()