import keras
import tensorflow as tf
from keras.datasets import fashion_mnist
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import tag_constants
import numpy as np
"""
拼接模型，成功
参考 https://blog.csdn.net/mogoweb/article/details/83064819
"""
num_classes=10
def getFashionmist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    # Input image dimensions.
    input_shape = x_train.shape[1:]
    print("input_shape", input_shape)
    # Normalize data.

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert class vectors to binary class matrices.
    y_train1 = keras.utils.to_categorical(y_train, num_classes)
    y_test1 = keras.utils.to_categorical(y_test, num_classes)
    #x_test1 = x_test
    #x_train1 = x_train
    return x_train,y_train1,y_train,x_test,y_test1,y_test

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
        ["Tnet","Tlogits"],
        variable_names_whitelist=None,
        variable_names_blacklist=None)
batchsize=128

with tf.Session() as sess:
    #原始输入
    x = tf.placeholder("float", [None, 28, 28, 1], name="X")
    #y是D的输出，T的输入
    y ,= tf.import_graph_def(g1def, input_map={"DX:0": x}, return_elements=["DY:0"])
    #z,z1 就是 net（输出的概率[batchsize,num of class]） logits (softmax前的输出[batchsize,num of class])
    z,z1= tf.import_graph_def(g2def, input_map={"TX:0": y}, return_elements=["Tnet:0","Tlogits:0"])
    tf.identity(z,name="net")
    tf.identity(z1, name="logits")
    """
    #保存模型
    tf.saved_model.simple_save(sess,
                               "./DTmodel",
                               inputs={"X": x},
                               outputs={"net": z,"logits": z1})
    
    """
    #测试准确率
    x_train, y_train1, y_train, x_test, y_test1, y_test=getFashionmist()
    xs=x_test[0:batchsize]
    err=0

    z11, z12 = sess.run([z, z1], feed_dict={'X:0': xs})
    netarg = np.argmax(z11, 1)
    for j in range(0, batchsize):
        xs0=x_test[j:j+1]
        print("j: ", j, " trulabel", y_test[j], "tprelabel", netarg[j])
        if y_test[j] != netarg[j]:
            err += 1.0
    print("准确率 ", (batchsize - err) / batchsize)