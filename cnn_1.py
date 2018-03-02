# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:44:03 2018

@author: Administrator
"""

import tensorflow as tf 
from gen_captcha import CHAR_SET_LEN, IMAGE_HEIGHT, IMAGE_WIDTH, MAX_CAPTCHA

X = tf.placeholder(tf.float32,[None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32,[None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)

def highway_conv2d_layer(input_x,filters_num):
    H = tf.layers.conv2d(inputs=input_x,filters=filters_num,kernel_size=[3,3],padding="same",activation = tf.nn.relu,kernel_initializer=tf.random_normal_initializer())
    T = tf.layers.conv2d(inputs=input_x,filters=filters_num,kernel_size=[3,3],padding="same",activation = tf.sigmoid,kernel_initializer=tf.random_normal_initializer())
    C = 1.0-T
    return H*T + input_x* C#tf.add(tf.matmul(H,T),tf.matmul(input_x,C))
    
    

#定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha = 0.1):
    x  = tf.reshape(X,[-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])
   
    
    #3 CONV layer
    
    
    """
    conv1 = tf.layers.conv2d(inputs=x,filters=32,kernel_size=[3,3],padding="same",activation = tf.nn.relu,kernel_initializer=tf.random_normal_initializer())
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size = [2,2],strides = 2)
    
    
    conv2 = tf.layers.conv2d(inputs = pool1, filters =64, kernel_size = [3,3],padding ="same",activation = tf.nn.relu,kernel_initializer=tf.random_normal_initializer())
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size =[2,2],strides = 2) 
    
    conv3 = tf.layers.conv2d(inputs = pool2,filters = 64,kernel_size =[3,3],padding = "same",activation = tf.nn.relu,kernel_initializer=tf.random_normal_initializer())
    pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size = [2,2],strides = 2)
    """
    #higuway
    conv1 = highway_conv2d_layer(x,64)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size = [2,2],strides = 2)
    conv2 = highway_conv2d_layer(pool1,64)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size =[2,2],strides = 2)
    conv3 = highway_conv2d_layer(pool2,64)
    pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size = [2,2],strides = 2)
    
    
    ##全连接
    
    pool2_flat = tf.reshape(pool3,[-1,7*20*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation = tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate = 0.5)
    
    
    
    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dropout, w_out), b_out)
    return  out 
    
    
    
    
    
    