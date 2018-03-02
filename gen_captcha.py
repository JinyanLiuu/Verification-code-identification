# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:23:05 2018

@author: Administrator
"""

import random
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha  # pip install captcha
import uuid

number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
gen_char_set = number + alphabet + ALPHABET
CHAR_SET_LEN = len(gen_char_set)
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4

def random_captcha_text(char_set=gen_char_set, captcha_size=MAX_CAPTCHA):
    """
    生成随机字符串
    """
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image():
    """
    生成图像大小(60, 160, 3)的验证码
    """
    while True: 
        image = ImageCaptcha()
        captcha_text = random_captcha_text()
        captcha_text = ''.join(captcha_text)
        captcha = image.generate(captcha_text)
        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)
        if captcha_image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            continue
        return captcha_text, captcha_image


def gen_and_save_image(num=50000):
    """
    批量生成验证图片集
    """
    for i in range(num):
        text, image = gen_captcha_text_and_image()
        im = Image.fromarray(image)
        uid = uuid.uuid1().hex
        image_name = '__%s__%s.png' % (text, uid)
        image_file = os.path.join(os.path.join(os.getcwd(), 'images'), image_name)
        im.save(image_file)


def demo_show_img():
    """
    使用matplotlib来显示生成的图片
    """
    text, image = gen_captcha_text_and_image()
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    #gen_and_save_image()
    #demo_show_img()
    pass