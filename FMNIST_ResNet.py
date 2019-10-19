import keras
import tensorflow as tf

import myDataset
import ResNet
import ResNet_dila
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import fashion_mnist


import os
def set_config():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True
    return gpuConfig


num_classes=10
MODEL_SAVE_PATH = "./model2/"
MODEL_NAME = "minst_resnet50_model.ckpt"
epochs=100
batchsize=100





def getFashionmist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    y_train1 = keras.utils.to_categorical(y_train, num_classes)
    y_test1 = keras.utils.to_categorical(y_test, num_classes)

    return x_train,y_train1,y_train,x_test,y_test1,y_test

def train_resnet502(x_train,y_train1):
    """
    训练T模型，并保存
    :param x_train: 训练样本输入
    :param y_train1: 训练样本真实输出
    :return:
    """
    batch_size = 128

    X = tf.placeholder("float", [None, 28, 28, 1],name="TX")
    Y = tf.placeholder("float", [None, 10],name="TY")
    learning_rate = tf.placeholder("float", [])
    global_step = tf.Variable(0, trainable=False)

    net,logits = ResNet.resnet_50(X,1)
    net_ = tf.identity(net, name='Tnet')
    logits_ = tf.identity(logits, name='Tlogits')
    cross_entropy = -tf.reduce_sum(Y * tf.log(net))
    opt = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(cross_entropy,global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    cfg=set_config()
    sess = tf.Session(config=cfg)
    init=tf.global_variables_initializer()
    sess.run(init)

    for i in range(0,epochs):
        l=x_train.shape[0]/batch_size
        for j in range(0,int(l)):
            start=j*batch_size
            end=(j+1)*batch_size
            xs=x_train[start:end]
            ys=y_train1[start:end]
            #print("xs.type",type(xs))
            _, loss_value, step, acc = sess.run([train_op, cross_entropy, global_step, accuracy],
                                                feed_dict={X: xs, Y: ys, learning_rate: 0.0003})
            print("Epoch %d Iter %d : loss on training "
                  "batch is %g . and accuracy is %g." % (i, j,loss_value, acc))
        # 每1000轮保存一次模型
        if i % 1 == 0:
            # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失
            # 函数大小。通过损失函数的大小可以大概了解训练的情况。在验证数
            # 据集上正确率的信息会有一个单独的程序来生成
            print("Epoch %d: 保存模型"% i)
            # 保存当前的模型。注意这里给出了global_step参数，这样可以让每个
            # 被保存的模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000”，
            # 表示训练1000轮之后得到的模型。
            #saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)

    sess.close()


def train_resnet50(dataset):
    batch_size = 128

    X = tf.placeholder("float", [None, 28, 28, 1],name="TX")
    Y = tf.placeholder("float", [None, 10],name="TY")
    learning_rate = tf.placeholder("float", [])
    global_step = tf.Variable(0, trainable=False)
    # ResNet Models
    net ,logits= ResNet.resnet_50(X,1)
    net_ = tf.identity(net, name='Tnet')
    logits_ = tf.identity(logits, name='Tlogits')
    cross_entropy = -tf.reduce_sum(Y * tf.log(net))
    opt = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(cross_entropy,global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    cfg = set_config()
    sess = tf.Session(config=cfg)
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        #print(i)
        xs, ys = dataset.next_batch(batch_size)
        #print("xs shape",xs.shape)
        #xs=tf.reshape(xs,[batch_size,28,28,1])
        #ys = tf.reshape(ys, [batch_size, 10])
        _, loss_value, step ,acc= sess.run([train_op, cross_entropy, global_step,accuracy],
                                       feed_dict={X: xs, Y: ys,learning_rate: 0.0003})
        print("Epoch %d , loss "
              " is %g.and accuracy is %g." % (i, loss_value, acc))
        # 每100轮保存一次模型
        if i % 50 == 0:
            tf.saved_model.simple_save(sess,
                                       "./newmodel1/" + str(i) + "/",
                                       inputs={"TX": X, "TY": Y},
                                       outputs={"Tnet": net_, "Tlogits": logits_})
    sess.close()

def test_resnet50DIY(x_test,y_test1):
    batch_size = 100
    numsap=x_test.shape[0]
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], "./newmodel1/100/")
        graph = tf.get_default_graph()
        l=1#numsap/batch_size
        totalacc=0
        for i in range(0,int(l)):
            start=i*batch_size
            end=(i+1)*batch_size
            xs, ys = x_test[start:end], y_test1[start:end]
            TX = sess.graph.get_tensor_by_name('TX:0')
            net=sess.graph.get_tensor_by_name('Tnet:0')
            err=0
            Yarg=np.argmax(ys,1)
            """
            net1 = sess.run(net, feed_dict={TX: xs})
            netarg = np.argmax(net1, 1)
            for j in range(0, batch_size):
                if Yarg[j] != netarg[j]:
                    err+=1.0
                print("truelabel",Yarg[j],"  prelabel ",netarg[j])
            """
            for j in range(0, batch_size):
                xs0=xs[j:j+1]
                net1 = sess.run(net, feed_dict={TX: xs0})
                netarg = np.argmax(net1, 1)
                if Yarg[j] != netarg[0]:
                    err+=1.0
                print("truelabel",Yarg[j],"  prelabel ",netarg[0], netarg)

            print("acc ",(batch_size-err)/batch_size)

            totalacc+=(batch_size-err)/batch_size
        print("准确率",totalacc/l)


def test_resnet50(dataset):
    batch_size = 128

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], "./newmodel/0/")
        graph = tf.get_default_graph()
        l=int(10000/batch_size)
        totalacc=0
        for i in range(0,l):
            xs, ys = dataset.next_batch(batch_size)
            TX = sess.graph.get_tensor_by_name('TX:0')
            logits = sess.graph.get_tensor_by_name('Tlogits:0')
            net = sess.graph.get_tensor_by_name('Tnet:0')
            #correctpre=np.equal()
            err = 0
            Yarg = np.argmax(ys, 1)

            net1 = sess.run(net, feed_dict={TX: xs})
            netarg = np.argmax(net1, 1)
            for j in range(0, batch_size):
                if Yarg[j] != netarg[0]:
                    err += 1.0
            acc=(batch_size - err) / batch_size
            totalacc+=acc
            print("准确率 ", acc)
        print("平均准确率 ", totalacc/l)

def main(argv=None):
    print("begin:")
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

    print("loadDataFinish,begin train....")
    x_train, y_train1, y_train, x_test, y_test1, y_test=getFashionmist()
    dataset=myDataset.Dataset(x_train,y_train1)
    #train_resnet50(dataset)#./model
    #x_train=x_train.astype('float32') / 255
    train_resnet50(dataset)#./model1

    dataset = myDataset.Dataset(x_test, y_test1)
    #print("loadDataFinish,begin test...")
    test_resnet50(dataset)

main()