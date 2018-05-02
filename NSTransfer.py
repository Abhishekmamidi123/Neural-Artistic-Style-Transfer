import os
import sys
import scipy.io
import scipy.misc
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy import io as sio
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from scipy.misc import imsave
import warnings
warnings.filterwarnings('ignore') # Ignores warnings.

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H*n_W,n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H*n_W,n_C]))
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))/(4*n_H*n_W*n_C)    
    return J_content

def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]))
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS,GG)))/(4 * n_C**2 * (n_W * n_H)**2)
    return J_style_layer

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha*J_content + beta*J_style
    return J

def generate_noise_image(content_image, noise_ratio, image_height, image_width, channels):
    noise_image = np.random.uniform(-20, 20, (1, image_height, image_width, channels)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image

image_height = 300
image_width = 400
channels = 3 # RGB

# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()
content_image = scipy.misc.imread("images/1/content.png")
content_image = np.array([misc.imresize(content_image, (300, 400))])
# content_image = reshape_and_normalize_image(content_image)
style_image = scipy.misc.imread("images/1/content.png")
style_image = np.array([misc.imresize(style_image, (300, 400))])
# style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image, 0.6, image_height, image_width, channels)
# imshow(generated_image[0])

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

# Create tensors for each layer.
model = {}
model['input'] = tf.Variable(np.zeros((1, image_height, image_width, channels)), dtype = 'float32')

model['conv1_1'] = tf.nn.relu(conv2d(model['input'], 0))
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
# print model

# model = load_vgg_model("../pretrained-model/imagenet-vgg-verydeep-19.mat") 
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
# Compute the content cost
J_content_1 = compute_content_cost(a_C, a_G)

out = model['conv3_2']
a_C = sess.run(out)
a_G = out
J_content_2 = compute_content_cost(a_C, a_G)
J_content = J_content_1 + J_content_2

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)
J = total_cost(J_content, J_style, alpha = 10, beta = 40)
optimizer = tf.train.AdamOptimizer(1)
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 3000):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    for i in range(num_iterations):
        _ = sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i%1 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            imsave('out_nst_2/1_output_'+str(i)+'.png', generated_image[0])
            #save_image("output/" + str(i) + ".png", generated_image)
    imsave('out_nst_2/1_output_'+str(i)+'.png', generated_image[0])
#    save_image('output/generated_image.jpg', generated_image)
    return generated_image
    
model_nn(sess, generated_image)
