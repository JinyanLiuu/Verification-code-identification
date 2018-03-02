# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:03:55 2018

@author: Administrator
"""
import tensorflow as tf
from cnn_1 import crack_captcha_cnn, Y, keep_prob, X
from data_iter import get_next_batch
from gen_captcha import CHAR_SET_LEN, MAX_CAPTCHA
import time
from datetime import timedelta
import os

save_dir = os.path.join(os.getcwd(), 'checkpoints')
tensorboard_dir = 'tensorboard'

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
         # 最后一层用来分类的softmax和sigmoid有什么不同？
	 # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/sougou_cnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    start_time = time.time()
    best_acc_val = 0.0           # 最佳验证集准确率
    last_improved = 0            # 记录上一次提升批次
    #require_improvement = 1600   # 如果超过xx轮未提升，提前结束训练

    with tf.Session() as sess:
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            
            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                s, acc, losss = sess.run([merged_summary, accuracy, loss], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                # 将训练结果写入tensorboard scalar
                writer.add_summary(s, step)
                
                if acc > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc
                    last_improved = step
                    saver.save(sess=sess, save_path=save_dir)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Step: {0:>6}, Train Loss: {1:>6.2}, Val Loss: {2:>6.2}, Train Acc: {3:>7.2%}, Time: {4} {5}'
                print(msg.format(step, loss_, losss, acc, time_dif, improved_str))
                
                """
                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                break  # 跳出循环
            """
                
            step += 1

train_crack_captcha_cnn()