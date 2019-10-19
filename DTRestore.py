import keras
import tensorflow as tf
import numpy as np
from keras.datasets import fashion_mnist
from tensorflow.examples.tutorials.mnist import input_data
"""
拼接模型保存后，加载并进行预测
参考 https://blog.csdn.net/mogoweb/article/details/83064819
"""
num_classes=10

def getFashionmist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train1 = keras.utils.to_categorical(y_train, num_classes)
    y_test1 = keras.utils.to_categorical(y_test, num_classes)

    return x_train,y_train1,y_train,x_test,y_test1,y_test

x_train,y_train1,y_train,x_test,y_test1,y_test=getFashionmist()
batchsize=128
with tf.Session(graph=tf.Graph()) as sess:
    #sess.run(tf.global_variables_initializer())
    tf.saved_model.loader.load(sess, ["serve"], "./DTmodel")
    graph = tf.get_default_graph()

    x = sess.graph.get_tensor_by_name('X:0')
    net= sess.graph.get_tensor_by_name('net:0')
    logits= sess.graph.get_tensor_by_name('logits:0')
    batch_xs =x_test[0:batchsize]
    batch_ys =y_test1[0:batchsize]

    err = 0
    idx = []
    for j in range(0, batchsize):
        batch_xs0 = x_test[j:j + 1]
        net1, logits1 = sess.run([net, logits], feed_dict={x: batch_xs0})
        netarg = np.argmax(net1, 1)
        print("j: ", j, " trulabel", y_test[j], "tprelabel", netarg[0])
        if y_test[j] != netarg[0]:
            # print("****")
            err += 1.0

    print("acc ", (batchsize - err) / batchsize)


