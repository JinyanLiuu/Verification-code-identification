# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:29:46 2018

@author: Administrator
"""
import os
import numpy as np
import tensorflow as tf
from gen_captcha import gen_captcha_text_and_image
from gen_captcha import CHAR_SET_LEN, IMAGE_HEIGHT, IMAGE_WIDTH, MAX_CAPTCHA
import matplotlib.pyplot as plt
from cnn_1 import crack_captcha_cnn, X, keep_prob
from data_iter import convert2gray, vec2text


_, random_im = gen_captcha_text_and_image()


def crack_captcha(captcha_image=random_im):
    if captcha_image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        captcha_image = captcha_image.resize((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    plt.imshow(captcha_image)
    plt.show()
    
    # 定义预测计算图
    output = crack_captcha_cnn()
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=os.path.join(os.getcwd(), 'checkpoints'))
        
        captcha_image = convert2gray(captcha_image)
        captcha_image = captcha_image.flatten() / 255
        o, p = sess.run([output, predict], feed_dict={X: [captcha_image], keep_prob: 1})
        
        v = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        for i in p[0]:
            v1 = np.argwhere(p[0] == i)
            idx = v1 * CHAR_SET_LEN + i
            v[idx] = 1
    return vec2text(v)
    

if __name__ == '__main__':
    a = crack_captcha()
    print(a)
