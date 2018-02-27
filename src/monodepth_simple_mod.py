# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
import cv2
import time

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *
from ThesisMasterNode import *

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--image_path',       type=str,   help='path to the image', default="/home/nvidia/monodepth-master/")
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', default="/home/nvidia/monodepth-master/")
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)

args = parser.parse_args()



def image_callback(data):
    # Simply print out values in our custom message.
    global image
    image = data

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def test_simple(params):
    """Test function."""

    left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
    model = MonodepthModel(params, "test", left, None)
    
    #t0=time.time()
    input_image = scipy.misc.imread(args.image_path, mode="RGB")
    original_height, original_width, num_channels = input_image.shape
    width=512
    height=256
    input_image=cv2.resize(input_image,(width,height))
    #input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)
    #thisimage=tf.Variable(input_images, name='thisimage')

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    time.sleep(5)

	
	
	
    while 1==1
		
		rospy.Subscriber("image", Image, self.image_callback)
		
		t0=time.time()
		input_image = image
		original_height, original_width, num_channels = input_image.shape
		width=512
		height=256
		input_image=cv2.resize(input_image,(width,height))
		#input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
		input_image = input_image.astype(np.float32) / 255
		input_images = np.stack((input_image, np.fliplr(input_image)), 0)
		t1=time.time()
		total=t1-t0 
		#print(total)




		disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
		disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

		t1=time.time()
		total=t1-t0 
		#print(total)

		output_directory = os.path.dirname(args.image_path)
		output_name = os.path.splitext(os.path.basename(args.image_path))[0]

		#np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
		disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [256, 512])

		imgnum = np.matrix(disp_to_img)
		imgnum = np.array(imgnum.sum(axis=0))
		dec = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
		dec[0] = np.sum(imgnum[0][4:46])
		dec[1] = np.sum(imgnum[0][46:91])
		dec[2] = np.sum(imgnum[0][91:136])
		dec[3] = np.sum(imgnum[0][136:181])
		dec[4] = np.sum(imgnum[0][181:226])
		dec[5] = np.sum(imgnum[0][226:271])
		dec[6] = np.sum(imgnum[0][271:316])
		dec[7] = np.sum(imgnum[0][316:361])
		dec[8] = np.sum(imgnum[0][361:406])
		dec[9] = np.sum(imgnum[0][406:451])
		dec[10] = np.sum(imgnum[0][451:496])

		#plt.plot(dec)
		#plt.ylabel('cost')
		#plt.xlabel('direction')
		#plt.show()




		#print (dec)
  
		#t1=time.time()
		#total=t1-t0 
		#print(total)

		#plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')
    
		#t1=time.time()
		#total=t1-t0 
		#print(total)

		#print('done!')

		ThesisMasterNode(dec)
		
def main(_):

    params = monodepth_parameters(
        encoder='vgg',
        height=256,
        width=512,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)



    test_simple(params)

if __name__ == '__main__':
    tf.app.run()
