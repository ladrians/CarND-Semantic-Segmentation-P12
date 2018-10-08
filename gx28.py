import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Added imports
import time
import scipy
from moviepy.editor import VideoFileClip
import numpy as np

# General Parameters
num_classes = 2
image_shape = (160, 576)
data_dir = './data'
runs_dir = './runs'
kernel_initializer = 1e-2 # 2e-2
kernel_regularizer = 1e-3 # 1e-2, 1e-3, 1e-4, 1e-5
kernel_size = 4
strides = (2, 2)
padding='same'
k_prob = 0.95 # 0.8, 0.9, 1.0
l_rate = 1e-3 # 1e-2, 1e-3, 1e-4
epochs = 20 # 1, 2, 10, 20
batch_size = 5 # 1, 2, 4, 5
input_file = 'C:/Proyectos/donkey/donkey/d2/tub_1_18-05-25.mp4'
output_file = 'C:/temp/gx/gx28/uav/tub_1_18-05-25_SemanticSegmentation.mp4'

def debug(layer, shape):
    ''' Debugging '''
    tf.Print(layer, shape)

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    t_vgg_input_tensor_name = graph.get_tensor_by_name(vgg_input_tensor_name)
    t_vvgg_keep_prob_tensor_name = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    t_vgg_layer3_out_tensor_name = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    t_vgg_layer4_out_tensor_name = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    t_vgg_layer7_out_tensor_name = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return t_vgg_input_tensor_name, t_vvgg_keep_prob_tensor_name, t_vgg_layer3_out_tensor_name, t_vgg_layer4_out_tensor_name, t_vgg_layer7_out_tensor_name
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """    

    # Added to skip ResourceExhaustedError, based on https://medium.com/@subodh.malgonde/transfer-learning-using-tensorflow-52a4f6bcde3e
    vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
    
    # Convolutions
    conv7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding=padding,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_regularizer),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=kernel_initializer))
    conv4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding=padding,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_regularizer),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=kernel_initializer))
    conv3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding=padding,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_regularizer),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=kernel_initializer))

    # Transpose, Upsample by 2, by 2 and by 8
    output_1 = tf.layers.conv2d_transpose(conv7_1x1, num_classes, kernel_size, strides, padding=padding,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_regularizer),
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=kernel_initializer))

    # Skip layers, add them in
    output_1 = tf.add(output_1, conv4_1x1, name='l_add_2')
    output_2 = tf.layers.conv2d_transpose(output_1, num_classes, kernel_size, strides=(2, 2), padding=padding,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_regularizer),
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=kernel_initializer))
    
    output_2 = tf.add(output_2, conv3_1x1, name='l_add_3')
    output = tf.layers.conv2d_transpose(output_2, num_classes, kernel_size*4, strides=(8, 8), padding=padding,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_regularizer),
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=kernel_initializer))

    ## TODO, not sure how to use it
    ## from https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100
    #pool3_out_scaled = tf.multiply(pool3_out, 0.0001, name='pool3_out_scaled')
    #pool4_out_scaled = tf.multiply(pool4_out, 0.01, name='pool4_out_scaled')

    #debug(output, [tf.shape(output)[1:3]])
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # 2D reshape
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Classification and Loss
    
    # Loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    
    '''
    When adding l2-regularization, setting a regularizer in the arguments of 
    the tf.layers is not enough. Regularization loss terms must be manually 
    added to your loss function. otherwise regularization is not implemented.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cross_entropy_loss = tf.add(cross_entropy_loss, sum(regularization_losses))
    '''
    
    # Training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
                                        
    ## Debugging
    #debug(logits, [logits])
    #debug(correct_label, [correct_label])

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """    
    
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess.run(init_g)
    sess.run(init_l)
    for epoch in range(epochs):
        i = 0
        for image, label in get_batches_fn(batch_size): # pair of images and labels
            _, c_entropy_loss = sess.run([train_op,cross_entropy_loss]
			,feed_dict={input_image:image, correct_label:label, keep_prob:k_prob, learning_rate:l_rate})
            i += 1
            if i % 10 == 0:
                print("Epoch {} Batch {} Loss {:.3f}".format(epoch+1, i, c_entropy_loss))
tests.test_train_nn(train_nn)

def run():
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Memory parameters https://medium.com/@lisulimowicz/tensorflow-cpus-and-gpus-configuration-9c223436d4ef
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        
        tf.global_variables_initializer()
        tf.local_variables_initializer()
        
        # Placeholders
        labels = tf.placeholder(tf.float32, shape = [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output, labels, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, labels, keep_prob, learning_rate)
        
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # Apply the trained model to a video
        def complete_pipeline(img):
            """
            Sample code taken from gen_test_output helper method
            """
            image = scipy.misc.imresize(img, image_shape)
            im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, input_image: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)
            return np.array(street_im)
        
        def video_pipeline(current_image):
            ''' Complete video pipeline '''
            return complete_pipeline(current_image)

        def generate_video(output, process_image):
            ''' Generate a video '''
            print('Generating video to: {}'.format(output))
            clip1 = VideoFileClip(input_file)#.subclip(1,10) # 0,5  TODO remove this
            video_clip = clip1.fl_image(process_image)
            video_clip.write_videofile(output, audio=False)
            clip1.reader.close()
            return output

        video_output = generate_video(output_file, video_pipeline)

if __name__ == '__main__':
    then = time.time()
    run()
    now = time.time()
    diff = now - then
    minutes, seconds = int(diff // 60), int(diff % 60)
    print('Elapsed time {:d}:{:d} minutes'.format(minutes, seconds))
