import numpy as np
import tensorflow as tf

ker = np.loadtxt('data/ker.csv')
im = np.loadtxt('data/im.csv')

img = tf.reshape(im,shape=[1,6,6,1])
kernel = tf.reshape(ker,shape=[3,3,1,1])

k1 = tf.Variable(kernel,name='k1')

conv1 = tf.nn.conv2d(img, k1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
relu1 = tf.nn.relu(conv1)
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

grad_pr = tf.gradients(pool1,conv1)
grad_rc = tf.gradients(relu1,conv1)
grad_ck = tf.gradients(conv1,k1)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(pool1))
    g = sess.run(grad_ck)[0]
    print(g.shape,'\n',np.reshape(g,(3,3)))