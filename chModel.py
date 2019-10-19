"""
A pure TensorFlow implementation of a convolutional neural network.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import tensorflow as tf

from cleverhans import initializers
from cleverhans.model import Model
import keras
import tensorflow as tf
from keras.datasets import fashion_mnist
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import tag_constants

"""
模型封装到类中，用于生成对抗样本
参考 https://github.com/tensorflow/cleverhans/blob/784fa58c46a49ec996ad34b77c2f64d08fb0a68a/cleverhans/model_zoo/basic_cnn.py#L18
"""

class ModelBasicCNN(Model):
  def __init__(self, scope, nb_classes, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())

    # Do a dummy run of fprop to make sure the variables are created from
    # the start
    #self.fprop(tf.placeholder(tf.float32, [None, 28, 28, 1],name="X"))
    # Put a reference to the params in self so that the params get pickled
    #self.params = self.get_params()
    #self.sess1=0
  def getSession(self):
      return self.sess1

  def fprop(self, x, **kwargs):
    """

    :param x: 输入的图片[batchsize,h,w,channel]
    :param kwargs:
    :return: logits : softmax 前的输出
             net ：最后预测结果[batchsize,num of class]
    """
    del kwargs
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
    #with tf.Graph().as_default() as gcombine: 会生成新图 去掉用的应该是默认图
    with tf.Session() as sess:
        #self.sess1 = sess
        #print("")
        x1=tf.placeholder("float", [None, 28, 28, 1], name="X")
        x1=x
        y, = tf.import_graph_def(g1def, input_map={"DX:0": x1}, return_elements=["DY:0"])
        net1,logits1 = tf.import_graph_def(g2def, input_map={"TX:0": y}, return_elements=["Tnet:0", "Tlogits:0"])

        return {self.O_LOGITS: logits1,
                self.O_PROBS: net1}

