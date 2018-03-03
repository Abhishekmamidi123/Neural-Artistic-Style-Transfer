
# coding: utf-8

# In[13]:


# README

# Tensorflow implementation of the paper: "A Neural Algorithm of Artistic Style"
# You can know more about the paper here: "https://arxiv.org/abs/1508.06576

# Inputs:
# 1. Content Image, 
# 2. Style Image

# Output:
# Generated Image(G)

# Unlike other optimization problems, we don't optimize the cost function to get weights. Here, we try to find the pixels of the 
# final(content+style) generated image(G) by reducing the error of the cost function.

# There are two cost functions:
# 1. J_content
# 2. J_style

# Total cost function: 
# J_total = alpha*J_content + beta*J_style

# J_content:
# 

# J_style:
# 

# VGG 19 model is taken from the paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# VGG 19 has 43 layers
# 0 is conv1_1 (3, 3, 3, 64)
# 1 is relu
# 2 is conv1_2 (3, 3, 64, 64)
# 3 is relu    
# 4 is avgpool
# 5 is conv2_1 (3, 3, 64, 128)
# 6 is relu
# 7 is conv2_2 (3, 3, 128, 128)
# 8 is relu
# 9 is avgpool
# 10 is conv3_1 (3, 3, 128, 256)
# 11 is relu
# 12 is conv3_2 (3, 3, 256, 256)
# 13 is relu
# 14 is conv3_3 (3, 3, 256, 256)
# 15 is relu
# 16 is conv3_4 (3, 3, 256, 256)
# 17 is relu
# 18 is avgpool
# 19 is conv4_1 (3, 3, 256, 512)
# 20 is relu
# 21 is conv4_2 (3, 3, 512, 512)
# 22 is relu
# 23 is conv4_3 (3, 3, 512, 512)
# 24 is relu
# 25 is conv4_4 (3, 3, 512, 512)
# 26 is relu
# 27 is avgpool
# 28 is conv5_1 (3, 3, 512, 512)
# 29 is relu
# 30 is conv5_2 (3, 3, 512, 512)
# 31 is relu
# 32 is conv5_3 (3, 3, 512, 512)
# 33 is relu
# 34 is conv5_4 (3, 3, 512, 512)
# 35 is relu
# 36 is avgpool
# 37 is fullyconnected (7, 7, 512, 4096)
# 38 is relu
# 39 is fullyconnected (1, 1, 4096, 4096)
# 40 is relu
# 41 is fullyconnected (1, 1, 4096, 1000)
# 42 is softmax


# In[27]:


# Import modules

import numpy as np
import scipy
from scipy import io as sio
from scipy import ndimage, misc
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # Ignores warnings.


# In[28]:


# Reading VGG19 weights and converting each layer into tensors and storing them in a dictionary.

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


# In[63]:


# 
image_filepath = 'images/butterfly.jpg'
image = ndimage.imread(image_filepath, mode="RGB")
image_resized = np.array([misc.imresize(image, (300, 400))])
model["input_image"].assign(image_resized)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# sess.run(model["conv4_2"])
