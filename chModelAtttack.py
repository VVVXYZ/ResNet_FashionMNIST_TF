"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

import keras
import numpy as np
import tensorflow as tf


from cleverhans.loss import CrossEntropy
from cleverhans.dataset import MNIST
from cleverhans.utils_tf import model_eval
from cleverhans.train import train
from cleverhans.attacks import FastGradientMethod,DeepFool,CarliniWagnerL2
from cleverhans.utils import AccuracyReport, set_log_level
from keras.datasets import fashion_mnist

from chModel import ModelBasicCNN
import os

from myDataset import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""
https://github.com/tensorflow/cleverhans/blob/master/cleverhans_tutorials/mnist_tutorial_tf.py
"""

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64
nb_classes=10

def getFashionmist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train1 = keras.utils.to_categorical(y_train, nb_classes)
    y_test1 = keras.utils.to_categorical(y_test, nb_classes)
    return x_train,y_train1,y_train,x_test,y_test1,y_test

def attack():

    batch_size=300
    x_train, y_train1, y_train, x_test, y_test1, y_test=getFashionmist()
    config_args = {}
    model = ModelBasicCNN('mymodel', nb_classes)
    dataset = Dataset(x_test,y_test1)


    fgsm_params = {
        'eps': 5,#步长，论文中8/16，设置8/16运行慢，调成5
        'clip_min': 0.,
        'clip_max': 1.
    }

    eval_params = {'batch_size': batch_size}
    x = tf.placeholder("float", [None, 28, 28, 1], name="X")
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))


    sess = tf.Session(config=tf.ConfigProto(**config_args))
    oriacct=0.0
    advacct=0.0
    l=10
    for i in range(0,l):
        print("---%d---" % i)
        x_set,y_set=dataset.next_batch(batch_size)#dataset是仿照自带的nextbatch写的，每次返回一批图片[bachtsize,h,w,c]
        #x_set=xs=tf.reshape(x_set,[None,28,28,1])
        preds = model.get_logits(x)#preds 返回的是图片预测概率，[bachsize，num of class]
        oriacc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
        print("原始图片准确率", oriacc)
        oriacct+=oriacc

        fgsm = FastGradientMethod(model, sess=sess)#生成FGSM攻击模型
        adv_x = fgsm.generate(x, **fgsm_params)#生成对抗样本
        #其他攻击类似
        preds_adv = model.get_logits(adv_x)
        advacc = model_eval(sess, x, y, preds_adv, x_set, y_set, args=eval_params)
        advacct+=advacc
        print("对抗样本准确率",advacc)

    print("advacc", advacct/l)
    print("oriacc", oriacct/l)

def main(argv=None):
    print("attack")
    attack()

if __name__ == '__main__':
    main()

