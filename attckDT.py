import foolbox
import keras
import tensorflow as tf
import numpy as np
from foolbox.distances import Linf
from keras.datasets import fashion_mnist
num_classes=10

def getFashionmist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    input_shape = x_train.shape[1:]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test=getFashionmist()
idx1=[59, 71, 85, 88, 96, 113, 120, 121,19, 27, 35,  125]
xt=0
yt=0
xt=x_test[0:len(idx1)]
yt=[]
batchsize=len(idx1)

for i in range(0,len(idx1)):
    xt[i]= x_test[idx1[i]]
    yt.append( y_test[idx1[i]])

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], "./DTmodel")
    graph = tf.get_default_graph()
    x = sess.graph.get_tensor_by_name('X:0')
    logits= sess.graph.get_tensor_by_name('logits:0')
    err=0.0
    with foolbox.models.TensorFlowModel(x, logits, (0, 255)) as model:
        for j in range(0, batchsize):
            image = xt[i]
            attack = foolbox.attacks.FGSM(model, distance=Linf, threshold=8)
            adversarial = attack(image, yt[i])
            print("trulabel ", yt[i], " prelabel ", np.argmax(model.predictions(image)), " advprelabel ",
                  np.argmax(model.predictions(adversarial)))


