import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.layers as tcl
import time
# from speech_data import data_reader
# import config
import matplotlib.pyplot as plt
import os
import logging
def create_valid_summary(dev_loss):
    values = [
        tf.Summary.Value(tag='dev_loss', simple_value=dev_loss)
    ]
    summary = tf.Summary(value=values)
    return summary
def create_folders(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
class MetricChecker(object):
    def __init__(self, cfg, less=True):
        self.early_stop_count = cfg.early_stop_count
        
        self.cur_dev = tf.placeholder(tf.float32, shape=[], name='cur_dev')
        if not less:
            self.best_dev = tf.get_variable(name='best_dev', trainable=False, shape=[],
                                            initializer=tf.constant_initializer(-np.inf))
            self.dev_improved = tf.less(self.best_dev, self.cur_dev)
        else:
            self.best_dev = tf.get_variable(name='best_dev', trainable=False, shape=[],
                                            initializer=tf.constant_initializer(np.inf))
            self.dev_improved = tf.less(self.cur_dev, self.best_dev)
        with tf.control_dependencies([self.dev_improved]):
            if not less:
                self.update_best_dev = tf.assign(self.best_dev,
                                                 tf.maximum(self.cur_dev, self.best_dev))
            else:
                self.update_best_dev = tf.assign(self.best_dev,
                                                 tf.minimum(self.cur_dev, self.best_dev))
        self.reset_step()

    def reset_step(self):
        self.stop_step = 0
    def reset_vir(self,sess):
        updata_vir = tf.assign(self.best_dev,100)
        opt = sess.run(updata_vir)

        # self.aa = 1

    def update(self, sess, cur_dev):
        dev_improved, best_dev = sess.run([self.dev_improved, self.update_best_dev],
                                          feed_dict={self.cur_dev: cur_dev})
        if dev_improved:
            self.reset_step()
        else:
            self.stop_step += 1
        return dev_improved, best_dev

    def should_stop(self):
        return self.stop_step >= self.early_stop_count

    def get_best(self, sess):
        return sess.run(self.best_dev)
def average_gradients(tower_grads, clip_grad):
    average_grads = []
    for grad_and_vars  in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expand_g = tf.expand_dims(g, axis=0)
            grads.append(expand_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)
        grad = tf.clip_by_norm(grad, clip_grad)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads