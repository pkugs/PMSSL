import os
import sys
project_path = os.path.abspath('..')
sys.path.append(project_path)

import logging
import time
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from matplotlib import pyplot as plt
from tools import *
# from cell import ConvLSTMCell
import pdb
from tensorflow.signal import stft as tf_stft
from tensorflow.signal import inverse_stft as tf_istft
logging.getLogger().setLevel(logging.INFO)
def l2_norm(s1, s2):
    norm = tf.reduce_sum(s1 * s2, axis=-1, keep_dims=True)
    return norm
def conv2d(x,W,strides):
    return tf.nn.conv2d(x,W,strides,padding='SAME')
def max_pool(x,m,n):
    return tf.nn.max_pool(x,ksize=[1,m,n,1],strides=[1,m,n,1],padding='SAME')
def get_weight(shape,lamda):
    initial=tf.truncated_normal(shape,stddev=0.1)
    var = tf.Variable(initial)
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(lamda)(var))
    return var
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def mu_law_encode(sig,ch):
    with tf.name_scope('encode'):
        mu = tf.to_float(ch-1)
        safe_audio_abs = tf.minimum(tf.abs(sig),1.0)
        mag = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(sig) * mag
        return signal
def mu_law_decode(sig,ch):
    with tf.name_scope('decode'):
        mu = ch-1
        mag = (1/mu) * ((1+mu)**abs(sig)-1)
        return tf.sign(sig) * mag
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10,dtype = numerator.dtype))
    return numerator / denominator
class ConvLstmModel(object):
    def __init__(self, sess, config, num_gpu,folder, initializer=None):
        self.session = sess
        self.config = config
        self.num_gpu = num_gpu
        self.epoch_counter = 0
        self.initializer = initializer
        self.eps = 1e-8
        self.global_step = tf.get_variable("global_step", shape=[], trainable=False,
                                           initializer=tf.constant_initializer(0),
                                           dtype=tf.int32)
        # define placeholder
        self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.create_placeholder()
        self.training = tf.placeholder(tf.bool, shape=[])
        # self.batch_size = tf.placeholder(tf.int32, shape=[])
        # init graph
        self.optimize()
        self.reset()

        train_event_dir = os.path.join(folder, 'train_event')
        dev_event_dir = os.path.join(folder, 'dev_event')
        create_folders(train_event_dir)
        create_folders(dev_event_dir)
        self.train_writer = tf.summary.FileWriter(train_event_dir, sess.graph)
        self.dev_writer = tf.summary.FileWriter(dev_event_dir)


    def create_placeholder(self):
        self._input = []
        self._shape = []
        self._pos=[]
        self._source=[]
        self._same_sign=[]
        self._input_batch=[]
        # self.batch_size = 10
        # send_len=self.config.send_len
        sig_len = self.config.sig_len
        hoa_dim = self.config.ref_ch
        pos_len=self.config.pos_len
        shape_len=self.config.shape_len
        ang_ch = self.config.ang_ch
        gamma_num = self.config.gamma_num
        one_hot_len = self.config.one_hot_len
        stft_len =self.config.stft_len
        time_len = int(np.ceil(sig_len*2/stft_len))+1
        freq_len = int(stft_len/2)+1
        for i in range(self.num_gpu):
            self._input.append(tf.placeholder(tf.float32, shape=[None, freq_len,time_len,hoa_dim*2]))
            self._source.append(tf.placeholder(tf.float32, shape=[None, freq_len,time_len,2]))
            self._input_batch.append(tf.placeholder(tf.int8, shape=[1]))

    def reset(self):
        self.batch_counter = 0
        self.total_loss = 0
        self.latest_loss = 0
        self.latest_loss1 = 0
        self.latest_loss2 = 0
        self.latest_loss3 = 0
        self.latest_batch_counter = 0
        self.epoch_counter += 1

    def Mul_Task_ConvLstmNet_new(self, inputs):
        # if self.training == True:
        batch_size = self.config.batch_size
        # stft_len =self.config.stft_len
        # else:
        #     batch_size = self.batch_size
        # shape_list = inputs.get_shape().as_list()
        # batch_size = shape_list[0]
        # print(batch_size)
        freq_len = self.config.stft_len
        time_shift = int(freq_len/2)
        hoa_dim = self.config.ref_ch
        keep_prob=self.config.keep_prob
        sig_len=self.config.sig_len
        time_step = int(np.ceil(sig_len*2/freq_len))+1
      

        ch_list1 = [32,32,32,64,128,256,512]
        kl_list1 = [5,3,3,3,3,3,3,3]
        h_dim_list = [3,3,3,3,3,3,3,3]
        stride1_list = [2,2,2,2,2,2,2,2]
        stride2_list = [1,1,1,1,1,1,1,1]
        

        encode_data = []
        input_len = []
        input_fre = []
        h_out  = inputs

        # h_out = tf.reshape(h_out,[batch_size,freq_len,time_step,hoa_dim*2])
        conv_w = int(freq_len/2)+1
        # h_out = self.channel_attention_v2(h_out)
        for l_ii in range(len(ch_list1)):
            with tf.variable_scope("conv_layer%d"%l_ii) as scope:
                input_ch = h_out.get_shape()[-1]
                input_len.append(tf.shape(h_out)[1])
                input_fre.append(tf.shape(h_out)[2])
                input_sig = h_out
                output_ch = ch_list1[l_ii]
                cw = weight_variable([kl_list1[l_ii],h_dim_list[l_ii],int(input_ch),output_ch])
                # cb = weight_variable([output_ch])
                # skip_pos = 2**l_ii
                h_out = tf.nn.conv2d(h_out,cw,1,padding='SAME')
                # h_out = tf.nn.atrous_conv2d(h_out,cw,skip_pos,'SAME')+cb
                h_out = max_pool(h_out,stride1_list[l_ii],stride2_list[l_ii])
                h_out=tf.contrib.layers.batch_norm(h_out,center=True, scale=True, scope='bn')
                h_out=tf.nn.dropout(h_out,keep_prob=keep_prob)
                conv_w = np.ceil((conv_w ) / stride1_list[l_ii]).astype(np.int32)
                # print(conv_w)

                # if l_ii<len(ch_list)-1:
                # if l_ii == 0:
                # # if l_ii == 0:
                #     h_out = tf.transpose(h_out,perm=[0,3,1,2])

                h_out = tf.nn.leaky_relu(h_out)
                # if l_ii<2:
                #     h_out = self.dense_net(h_out)

                # if l_ii < len(ch_list1)-1:
                encode_data.append(h_out)

        dim_len = tf.shape(h_out)[1]
        ch_len = tf.shape(h_out)[-1]
        h_out = tf.transpose(h_out,[0,2,1,3])
        rnn_dim = conv_w*ch_list1[-1]
        # print(h_out.shape)
        # print(time_step)
        # print(conv_w)
        # print(ch_list1[-1])
        # print(rnn_dim)
        rnn_input = tf.reshape(h_out,[batch_size,time_step,rnn_dim])
        rnn_type = 'lstm'
        hidden_size = [rnn_dim/2]
        # print(hidden_size)
        rnn_cell = tf.contrib.rnn.LSTMCell
        for rnn_ii in range(len(hidden_size)):
            with tf.variable_scope("{}_{}".format(rnn_type, rnn_ii)):

                fw_cell = rnn_cell(hidden_size[rnn_ii], use_peepholes=True)
                bw_cell = rnn_cell(hidden_size[rnn_ii], use_peepholes=True)
                initial_fw = fw_cell.zero_state(batch_size, dtype=tf.float32)
                initial_bw = bw_cell.zero_state(batch_size, dtype=tf.float32)
                output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, rnn_input,
                                                            # sequence_length=tf.to_int32(seq_len),
                                                            initial_state_fw=initial_fw,
                                                            initial_state_bw=initial_bw,
                                                            time_major=False, dtype=tf.float32)
                output = tf.concat(output, axis=2)
                rnn_input = output
        # outdims = hidden_size * 2
        reshape_out = tf.reshape(output, [batch_size,time_step, conv_w,ch_list1[-1]])
        h_out = tf.transpose(reshape_out,[0,2,1,3])


        for l_ii in range(len(ch_list1)):
            with tf.variable_scope("deconv_layer%d"%l_ii) as scope:
                
                # if l_ii > len(ch_list1)-3:
                #     h_out = self.dense_net(h_out)
                if l_ii == len(ch_list1)-1:
                    output_ch = 14
                else:
                    output_ch = ch_list1[len(ch_list1)-1-l_ii]
                # if l_ii < len(ch_list1)-1:
                if l_ii > 0:
                    # print(l_ii)
                    h_out = tf.concat([h_out,encode_data[len(ch_list1)-1-l_ii]],3)
                # print(h_out.shape)
                input_ch = h_out.get_shape()[-1]
                shape_inf = h_out.get_shape()
                kernel0 = tf.get_variable("kernel0", shape=[kl_list1[len(ch_list1)-1-l_ii],1,output_ch,input_ch])
                h_out = tf.nn.conv2d_transpose(h_out,kernel0,
                        output_shape = [batch_size,input_len[len(ch_list1)-1-l_ii],input_fre[len(ch_list1)-1-l_ii],output_ch],
                        strides = [1,stride1_list[len(ch_list1)-1-l_ii],stride2_list[len(ch_list1)-1-l_ii],1], padding = 'SAME')

                h_out=tf.contrib.layers.batch_norm(h_out,center=True, scale=True, scope='bn')
                h_out=tf.nn.dropout(h_out,keep_prob=keep_prob)

                # h_out = tf.nn.elu(h_out)
                if l_ii < len(ch_list1)-1:
                    h_out = tf.nn.leaky_relu(h_out)
                else:
                    h_out = tf.nn.sigmoid(h_out)


        
        # h_out
        # print(h_outSe_step)
        h_out = h_out*inputs
        h_out = tf.reshape(h_out,[batch_size*(time_shift+1)*time_step,14])
        h_out = tcl.fully_connected(inputs=h_out, num_outputs=2, activation_fn=None)
        h_out = tf.reshape(h_out,[batch_size,(time_shift+1),time_step,2])
        


        return h_out


    def stft_loss(self,source,pred):
        batch_size = self.config.batch_size
        freq_len = self.config.stft_len
        time_shift = int(freq_len/2)
        batch_sig = []
        pred_sig = []
        for batch_ii in range(batch_size):
            temp_sig = source[batch_ii,:]
            # print(temp_sig.type)
            temp_fft = tf_stft(signals=temp_sig,frame_length=freq_len,
                            frame_step=time_shift)
            temp_fft = tf.transpose(temp_fft,[1,0])
            real_part = tf.real(temp_fft,name=None)
            imag_part = tf.imag(temp_fft,name=None)
            stack_comp = tf.stack([real_part,imag_part],2)
            batch_sig.append(stack_comp/tf.reduce_max(stack_comp))
            pred_sig.append(pred[batch_ii]/tf.reduce_max(pred[batch_ii]))
        loss = tf.reduce_mean((batch_sig-pred)**2)
        return loss

    def si_snr(self,s1,s2, eps=1e-8):
        s1_s2_norm = l2_norm(s1, s2)
        s2_s2_norm = l2_norm(s2, s2)
        s_target = s1_s2_norm/(s2_s2_norm+eps)*s2
        e_noise = s1 - s_target
        target_norm = l2_norm(s_target, s_target)
        noise_norm = l2_norm(e_noise, e_noise)
        snr = 10 * tf.log((target_norm) / (noise_norm+eps)+eps)
        return tf.reduce_mean(snr)
    def amp_error(self,source,pred):
        source_ene = tf.sqrt(source[:,:,:,0]**2+source[:,:,:,1]**2)
        pred_ene = tf.sqrt(pred[:,:,:,0]**2+pred[:,:,:,1]**2)
        return tf.reduce_mean((source_ene-pred_ene)**2)
    def tower_cost(self, inputs,source):
        pred= self.Mul_Task_ConvLstmNet_new(inputs)

        batch_size = self.config.batch_size
        pred_len = tf.shape(pred)[1]

        loss3 = tf.reduce_mean((source-pred)**2)
        loss2 = self.amp_error(source,pred)
        # for batch_ii in range(batch_size):
        #     loss2+=self.si_snr(pred[batch_ii,512:pred_len-512],source[batch_ii,512:pred_len-512])
        # loss3 = self.stft_loss(source,pred)
        loss = loss2*1+loss3*1
        # loss31 = tf.reduce_mean((inputs[:,:,0]-source)**2)
        #         (logits=pred,labels=source))
        # loss = loss1*0+loss3
        # one_hot_est = tf.nn.sigmoid(pred[0])
        return loss,loss,pred


    def optimize(self):
        self.lr0=0.0001
        self.lr_decay=0.998
        self.lr_step=15
        self.lr=tf.train.exponential_decay(
            self.lr0,
            self.global_step,
            decay_steps=self.lr_step,
            decay_rate=self.lr_decay,
            staircase=True)
        optimizer = tf.train.AdamOptimizer(self.lr)
        tower_cost = []
        tower_loss3 = []
        tower_sig_out = []
        for i in range(self.num_gpu):
            worker = '/gpu:%d' % i
            device_setter = tf.train.replica_device_setter(
                worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
            with tf.variable_scope("Model", reuse=(i>0)):
                with tf.device(device_setter):
                    with tf.name_scope("tower_%d" % i) as scope:

                        cost,loss3,sig_out =\
                         self.tower_cost(self._input[i], self._source[i])
                        tower_cost.append(cost)
                        tower_sig_out.append(sig_out)
                        tower_loss3.append(loss3)
        self.avg_cost = tf.reduce_mean(tower_cost)
        self.avg_loss3 = tf.reduce_mean(tower_loss3)     
        self.tower_sig_out = tower_sig_out
        self.apply_gradients_op = optimizer.minimize(self.avg_cost,global_step=self.global_step)
        tf.summary.scalar('avg_cost', self.avg_cost)
        self.merged = tf.summary.merge_all()

    def run_batch(self, group_data):
        feed_dict = {self.training: True}
        # self._input_batch  = np.zeros((len(group_data[0])))
        for i in range(self.num_gpu):
            feed_dict[self._input[i]] = group_data[0][i]
            feed_dict[self._source[i]] = group_data[1][i]
            # feed_dict[self._input_batch[i]] = len(group_data[0][i])
        start_time = time.time()
        _, i_global,loss,lr,loss3,pred = self.session.run(
            [self.apply_gradients_op, self.global_step, self.avg_cost,self.lr,
             self.avg_loss3,self.tower_sig_out],feed_dict=feed_dict)
        self.total_loss += loss
        self.latest_loss3 += loss3
        self.batch_counter += 1
        self.latest_batch_counter += 1
        duration = time.time() - start_time
        out_loss3 = self.latest_loss3 / self.latest_batch_counter,
        if i_global % self.config.log_period == 0:
            logging.info("Epoch {:d}, Average Train MSE: {:.6f}={:.6f}/{:d}, "
                         "Latest MSE: {:.6f},L3: {:.6f}, Duration: {:.2f} sec, Lr {:,.6f}".format(
                         self.epoch_counter, self.total_loss / self.batch_counter,
                         self.total_loss, self.batch_counter,
                         (self.latest_loss3) / self.latest_batch_counter,
                         self.latest_loss3 / self.latest_batch_counter,
                        duration * self.config.log_period,lr))
            self.latest_batch_counter = 0
            self.latest_loss3 = 0
        return i_global,out_loss3,pred

    def get_pred(self, data):
        feed_dict = {self.training: False}
        # self.batch_size = len(data)
        for i in range(self.num_gpu):
            feed_dict[self._input[i]] =data
        sig_out= self.session.run(self.tower_sig_out,feed_dict=feed_dict)
        return sig_out

    def valid(self, reader):
        total_loss, batch_counter = 0.0, 0
        total_loss1, total_loss2, total_loss3 = 0.0, 0.0,0.0
        num_sent = 0
        logging.info("Start to dev")
        start_time = time.time()
        while True:
            group_data = reader.next_batch()
            if group_data == None:
                reader.reset()
                break
            else:
                feed_dict = {self.training: False}
                for i in range(self.num_gpu):
                    feed_dict[self._input[i]] = group_data[0][i]
                    feed_dict[self._source[i]] = group_data[1][i]

                loss,loss3= self.session.run([self.avg_cost,self.avg_loss3], feed_dict=feed_dict)
                total_loss3 += loss3
                total_loss =total_loss3
                batch_counter += 1
                if batch_counter % 10 == 0:
                    logging.info("Dev environment {:d}, AVG Dev MSE: {:.6f}={:.6f}/{:d}"
                                 "L3:{:.6f}, Speed: {:.2f} environment/sec".format(
                                 batch_counter, total_loss/batch_counter, total_loss,
                                 batch_counter, 
                                 total_loss3/batch_counter, batch_counter/(time.time()-start_time)))
        duration = time.time() - start_time
        avg_loss = total_loss3 / batch_counter
        dev_summary = create_valid_summary(avg_loss)
        i_global = self.session.run(self.global_step)
        self.dev_writer.add_summary(dev_summary, i_global)
        logging.info("Finish dev {:d} environment in {:.2f} seconds, "
                     "AVG MSE: {:.6f}".format(batch_counter, duration, avg_loss))
        return avg_loss

    def save_model(self, i_global):
        model_path = os.path.join(self.job_dir, 'model.ckpt')
        self.saver.save(self.session, model_path, global_step=i_global)
        logging.info("Saved model, global_step={}".format(i_global))

    def restore_model(self):
        load_option = self.config.load_option
        if load_option == 0:
            load_path = tf.train.latest_checkpoint(self.job_dir)
        elif load_option == 1:
            load_path = tf.train.latest_checkpoint(self.best_loss_dir)
        else:
            load_path = self.config.load_path
        try:
            self.saver.restore(self.session, load_path)
            logging.info("Loaded model from path {}".format(load_path))
        except Exception as e:
            logging.error("Failed to load model from {}".format(load_path))
            raise e
    def reset_loss(self,sess):
        self.lr0=0.0015
        self.lr_decay=0.999
        self.lr_step=15
        self.lr=tf.train.exponential_decay(
            self.lr0,
            self.global_step,
            decay_steps=self.lr_step,
            decay_rate=self.lr_decay,
            staircase=True)
        updata_step = tf.assign(self.global_step,0)
        opt = sess.run(updata_step)

