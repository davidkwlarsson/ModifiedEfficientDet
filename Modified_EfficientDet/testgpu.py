
import tensorflow as tf

# tf.test.is_gpu_available  
# tf.test.gpu_device_name

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# assert tf.test.is_gpu_available()
# assert tf.test.is_built_with_cuda()

# print(tf.test.is_built_with_cuda())

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    print(c)