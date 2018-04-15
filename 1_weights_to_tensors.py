import scipy
import cPickle
import numpy as np
import tensorflow as tf
from scipy import io as sio
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from scipy.misc import imsave
import warnings
warnings.filterwarnings('ignore') # Ignores warnings.

path_vgg19_weights = '../pretrained_models/imagenet-vgg-verydeep-19.mat'
vgg_model = sio.loadmat(path_vgg19_weights)
vgg_layers = vgg_model['layers']

def conv2d(previous_layer, layer):
    W = vgg_layers[0][layer][0][0][2][0][0]
    b = vgg_layers[0][layer][0][0][2][0][1]
    layer_name = vgg_layers[0][layer][0][0][0][0]
    convolution = tf.nn.conv2d(previous_layer, filter = tf.constant(W), strides = [1,1,1,1], padding = 'SAME')
    bias = tf.constant(np.reshape(b, b.size))
    return convolution + bias

image_height = 300
image_width = 400
channels = 3 # RGB

# Create tensors for each layer.
model = {}
model['input_image'] = tf.Variable(np.zeros((1, image_height, image_width, channels)), dtype = 'float32')

model['conv1_1'] = tf.nn.relu(conv2d(model['input_image'], 0))
model['conv1_2'] = tf.nn.relu(conv2d(model['conv1_1'], 2))
model['avgpool1'] = tf.nn.avg_pool(model['conv1_2'], ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

model['conv2_1'] = tf.nn.relu(conv2d(model['avgpool1'], 5))
model['conv2_2'] = tf.nn.relu(conv2d(model['conv2_1'], 7))
model['avgpool2'] = tf.nn.avg_pool(model['conv2_2'], ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

model['conv3_1'] = tf.nn.relu(conv2d(model['avgpool2'], 10))
model['conv3_2'] = tf.nn.relu(conv2d(model['conv3_1'], 12))
model['conv3_3'] = tf.nn.relu(conv2d(model['conv3_2'], 14))
model['conv3_4'] = tf.nn.relu(conv2d(model['conv3_3'], 16))
model['avgpool3'] = tf.nn.avg_pool(model['conv3_4'], ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

model['conv4_1'] = tf.nn.relu(conv2d(model['avgpool3'], 19))
model['conv4_2'] = tf.nn.relu(conv2d(model['conv4_1'], 21))
model['conv4_3'] = tf.nn.relu(conv2d(model['conv4_2'], 23))
model['conv4_4'] = tf.nn.relu(conv2d(model['conv4_3'], 25))
model['avgpool4'] = tf.nn.avg_pool(model['conv4_4'], ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

model['conv5_1'] = tf.nn.relu(conv2d(model['avgpool4'], 28))
model['conv5_2'] = tf.nn.relu(conv2d(model['conv5_1'], 30))
model['conv5_3'] = tf.nn.relu(conv2d(model['conv5_2'], 32))
model['conv5_4'] = tf.nn.relu(conv2d(model['conv5_3'], 34))
model['avgpool5'] = tf.nn.avg_pool(model['conv5_4'], ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print model
