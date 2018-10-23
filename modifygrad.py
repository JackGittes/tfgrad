import tensorflow as tf

with tf.variable_scope('grad'):
    fc_grad_w = tf.Variable(tf.constant(0.,shape=[80,4]))
    fc_grad_wb = tf.Variable(tf.constant(0.,shape=[4]))

with tf.variable_scope('params'):
    k1 = tf.Variable(tf.truncated_normal(shape=[3,3,1,5],stddev=0.2,mean=0),name='k1')
    b1 = tf.Variable(tf.constant(1.0,shape=[5],dtype=tf.float32),name='b1')

    k2 = tf.Variable(tf.truncated_normal(shape=[3,3,5,5],stddev=0.5,mean=0),name='k2')
    b2 = tf.Variable(tf.constant(1.0,shape=[5],dtype=tf.float32),'b2')

    w = tf.Variable(tf.truncated_normal(shape=[80,4],stddev=0.1,mean=0),name='w')
    wb = tf.Variable(tf.constant(1.0,shape=[4],dtype=tf.float32),'wb')

x = tf.placeholder(shape=[None,16,16,1],dtype=tf.float32)
y_ = tf.placeholder(shape=[None,4],dtype=tf.float32)
with tf.name_scope('net'):
    conv1 = tf.nn.conv2d(x,k1,strides=[1, 1, 1, 1],padding='SAME',name='conv1')
    relu1 = tf.nn.relu(conv1)
    pool1 =tf.nn.max_pool(relu1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')

    conv2 = tf.nn.conv2d(pool1,k2,strides=[1,1,1,1],padding='SAME',name='conv2')
    relu2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(relu2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool2')

    fc_in = tf.reshape(pool2,shape = [-1,4*4*5],name='flatten')
    fc = tf.add(tf.matmul(fc_in,w),wb)

with tf.name_scope('name'):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(fc,y_,name='loss')

    fc_grad_one = tf.gradients(loss,[w,wb])
    fc_grad_w = tf.add(fc_grad_one[0],fc_grad_w)
    fc_grad_wb = tf.add(fc_grad_one[1],fc_grad_wb)




