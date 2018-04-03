import scipy
import numpy as np
import tensorflow as tf
from scipy import io as sio
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
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

# Read content and style images
content_image = scipy.misc.imread("images/louvre.jpg")
content_image = np.array([misc.imresize(content_image, (300, 400))])
imshow(content_image[0])
print "Content Image:"
plt.show()

style_image = scipy.misc.imread("images/monet_800600.jpg")
style_image = np.array([misc.imresize(style_image, (300, 400))])
imshow(style_image[0])
print "Style Image:"
plt.show()

# Generate a noisy random image - Generated image
noise_image = np.random.uniform(-20, 20, size=(1, image_height, image_width, channels)) #again y??
generated_image = noise_image * 0.6 + content_image * (1 - 0.6) ##again y??
imshow(generated_image[0])
print "Noise Image:"
plt.show()

sess = tf.Session()

# Generated image
sess.run(model['input_image'].assign(generated_image))
layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1', 'conv4_2']
generated_activations = []
for layer in layers:
	generated_activations.append(model[layer])

# Content image
sess.run(model['input_image'].assign(content_image))
content_activations = sess.run(model['conv4_2'])

# Style image
sess.run(model['input_image'].assign(style_image))
style_activations = []
for layer in layers:
	style_activations(sess.run(model[layer]))
	 
# Content cost
J_content = tf.reduce_sum(tf.square(tf.subtract(content_activation, generated_activations[-1])))
J_content = J_content/2.0

# Style cost
J_style = 0
for layer in layers[:-1]:
	shape = style_activation.shape
	height = shape[1]
	width = shape[2]
	channels = shape[3]
    
    # Reshape style_activations
	print style_activation.shape
	style_activation = tf.reshape(style_activation, [height*width, channels])
	print style_activation.shape
	
	# Compute Gram matrix for style_activation
	style_gram_matrix = tf.matmul(tf.transpose(style_activation), style_activation)
	print style_gram_matrix.shape
	
	# Activations of generated image
	generated_activation = model[layer]
	
	# Reshape generated_activations
	generated_activation = tf.reshape(generated_activation, [height*width, channels])
	
	# Compute Gram matrix for generated_activation
	generated_gram_matrix = tf.matmul(tf.transpose(style_activation), style_activation)
	print generated_gram_matrix.shape
	
	M = height*width
	N = channels
	
	# Compute J_style for this layer and add.
	J_style_for_this_layer = (1/(4.0*(N**2)*(M**2))) * tf.reduce_sum(tf.square(tf.subtract(style_activation, generated_activation))) 
	J_style += (1.0/len(layers)) * J_style_for_this_layer
