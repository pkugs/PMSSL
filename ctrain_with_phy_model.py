import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.layers as tcl
import time
# from speech_data3 import data_reader
import config_new as cfg
import matplotlib.pyplot as plt
import os,sys
# from Fullyconnected import fullConnModel
# from ConvLstm4_6_t3 import ConvLstmModel
# from ConvLstm_doa_time_corr import ConvLstmModel as doa_net
from Unet_LSTM import ConvLstmModel as enhance_net
import logging
import random
from tools import MetricChecker
# from load_file_list import get_file_list
from scipy.io import loadmat,wavfile
from scipy.io import savemat
import soundfile as sf
from scipy.fftpack import fft, ifft
from scipy.signal import stft, istft
from scipy import signal
from get_location import get_locat
import math
import pdb


tf.logging.set_verbosity(tf.logging.ERROR)

def si_snr(source,est, eps=1e-18):
    # print(source.shape)
    # print(est.shape)
    s_est_norm = np.sum(source*est)
    s_s_norm = np.sum(source*source)
    est_source = s_est_norm/(s_s_norm+eps)*source
    est_noise = est - est_source
    target_norm = np.linalg.norm(est_source,2)
    noise_norm = np.linalg.norm(est_noise,2)
    snr = 20 * np.log10((target_norm) / (noise_norm+eps)+eps)
    return snr 

def add_noise(sig,snr):
	if len(sig.shape)>1:
		noise = np.random.randn(sig.shape[0],sig.shape[1])
	else:
		noise = np.random.randn(sig.shape[0])
	# noise = noise-np.mean(noise)

	# pdb.set_trace()
	sig_std = np.std(sig)
	noise_var = sig_std**2/np.power(10,(snr/10))
	noise = (np.sqrt(noise_var)/np.std(noise))*noise
	return sig+noise
def cart2sph(x,y,z):
	azimuth = np.arctan2(y,x)
	elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
	r = np.sqrt(x**2 + y**2 + z**2)
	return azimuth, elevation, r

def sph2cart(azimuth,elevation,r):
	x = r * np.cos(elevation) * np.cos(azimuth)
	y = r * np.cos(elevation) * np.sin(azimuth)
	z = r * np.sin(elevation)
	return x, y, z
def search_file(file_dir):
	L=[];
	for root,dirs,files in os.walk(file_dir):
		for file in files:
			if os.path.splitext(file)[1]=='.mat'or os.path.splitext(file)[1]=='.wav':
				L.append(os.path.join(root,file))
	return L
# def creat_mic_tf(fs,)
def creat_reverb_tf(SH_mat,mic_pos,source_pos,room_shape):
	c = 344						# Sound velocity (m/s)
	air_dens = 16.367
	fs = 16000					# Sample frequency (samples/s)
	beta = 0.6+random.random()*0.4					# Reflections Coefficients
	tf_len = 2000				# Number of samples
	mtype = 'omnidirectional'	# Type of microphone
	ref_order = 3   			# Reflection order
	dim = 3						# Room dimension
	orientation = 0				# Microphone orientation (rad)
	hp_filter = 1				# Enable high-pass filter
	sh_order = 4
	hoa_dim = (sh_order+1)**2
	# n = 5000					# Number of samples
	loc1 = get_locat(c, fs, mic_pos, source_pos, room_shape, 
			beta=beta, nsample=tf_len, mtype=mtype, dim=dim)
	loc1 = loc1[0]
	hoa_tf = np.zeros((tf_len, hoa_dim), dtype='float32')
	out_count = 0
	s_inf = np.zeros((7,3), dtype='float32')
	for ref_ii in range(6):
		img_pos = loc1[ref_ii]
		# print(img_pos)
		# pdb.set_trace()
		# print(img_pos.shape)
		for jj in range(len(img_pos)):
			s_xyz = np.squeeze(img_pos[jj])
			[azi,ele,dist] = cart2sph(s_xyz[0],s_xyz[1],s_xyz[2])
			# if ref_ii<2:
			# 	print(s_xyz)
			# azi = azi - math.pi
			# if azi < -math.pi:
			# 	azi = azi+2*math.pi
			# print(dist)
			# pdb.set_trace()
			point = int(np.round(dist/c*fs)-2)
			s_sig = np.zeros((tf_len),dtype='float32')
			if ref_ii == 0:
				s_sig[point] = air_dens/(4*math.pi*dist)
			if ref_ii < 2:
				s_inf[out_count,:]=[azi,ele,dist]
				out_count += 1


			azi_i = int(np.min([round(azi/math.pi*180)+180,359]))
			ele_i = int(np.min([round(ele/math.pi*180)+90,179]))
			hoa_coef = np.squeeze(SH_mat[azi_i,ele_i,:])	
			if point<tf_len:		
				hoa_tf[point, :] = hoa_tf[point,:] +\
							 hoa_coef*(beta**ref_ii)*air_dens/(4*math.pi*dist)
	return hoa_tf,s_inf
def creat_room_sig(signal,tf):
	sig_len = len(signal)+len(tf[:,0])-1
	ch_num = len(tf[0,:])
	out_sig = np.zeros((sig_len,ch_num),dtype='float32')
	for ch_ii in range(ch_num):
		out_sig[:,ch_ii] = np.convolve(signal,tf[:,ch_ii])
	return out_sig

# def total_process()


def creat_forward_model_sig(input_sig,s_inf,SH_mat,file_ii):
	c = 344
	fs = 16000	
	data = loadmat(file_ii)
	# data = data["data"][0][0]
	# mic_tf = data['mic_tf'][0]
	# TT = data['TT']
	# error_ind = cfg.error_ind
	# 
	room_shape = data["data"]['room_shape'][0][0].astype(np.float32)
	mic_pos = data["data"]['mic_pos'][0][0].astype(np.float32)
	source_pos = data["data"]['source_pos'][0][0].astype(np.float32)
	dir_sig = data['data']['dir_sig1'][0][0].astype(np.float32)
	hoa_tf = data['data']['hoa_tf'][0][0].astype(np.float32)
	s_inf = data['data']['s_inf1'][0][0].astype(np.float32)
	# mic_pos = room_shape-mic_pos
	# source_pos  =room_shape-source_pos
	mic_dim = 32

	file = sys.path[0]+'/more_wavs_0109/'
	file_list = search_file(file)

	room_shape = room_shape+(np.random.rand(3)-0.5)
	mic_pos = mic_pos+(np.random.rand(3)-0.5)
	source_pos = source_pos+(np.random.rand(3)-0.5)

	hoa_tf,s_inf = creat_reverb_tf(SH_mat,mic_pos,source_pos,room_shape)

	gpu_num = len(input_sig)

	sig_len = cfg.sig_len
	conv_sig_len = sig_len+len(hoa_tf[:,0])-1
	hoa_dim = len(hoa_tf[0,:])
	ref_ch = cfg.ref_ch
	stft_len = cfg.stft_len
	time_len = int(np.ceil(sig_len*2/stft_len))+1
	freq_len = int(stft_len/2)+1
	batch_size = cfg.batch_size
	order = cfg.order
	
	hoa_coef = np.zeros((ref_ch,hoa_dim))
	delay_pot = np.zeros((ref_ch),dtype='int')
	# creat_reverb_tf(SH_mat,mic_pos,source_pos,room_shape)
	for ch_ii in range(ref_ch):
		azi_i = int(np.min([round(s_inf[ch_ii,0]/math.pi*180)+180+(random.random()-0.5)*5,359]))
		ele_i = int(np.min([round(s_inf[ch_ii,1]/math.pi*180)+90+(random.random()-0.5)*5,179]))
		hoa_coef[ch_ii,:] = np.squeeze(SH_mat[azi_i,ele_i,0:hoa_dim])
		delay_pot[ch_ii] = int(np.round(s_inf[ch_ii,2]/c*fs)-2)
	# pdb.set_trace()
	group_ref_sig = []
	group_dir_sig = []
	for gpu_ii in range(gpu_num):
		batch_sig = input_sig[gpu_ii]
		batch_size = len(batch_sig)
		batch_align_ref = np.zeros((batch_size, freq_len,time_len, ref_ch*2), dtype=np.float32)
		batch_dir_sig = np.zeros((batch_size, freq_len,time_len, 2), dtype=np.float32)
		for batch_ii in range(batch_size):
			# pred_sig = batch_sig[batch_ii,:,:,:]
			temp_sign = random.random()
			if temp_sign<0.5:
				pred_sig = batch_sig[0,:,:,:]
				f_sig = pred_sig[:,:,0]+1j*pred_sig[:,:,1]
				_,t_source = istft(f_sig,fs=1,window='hann',nperseg = cfg.stft_len)
				out_source = t_source[0:sig_len]
				# out_source = add_noise(out_source,random.random()*20+10)
			else:

				_, data = wavfile.read(file_list[round(random.random()*1500)])
				while (len(data)<sig_len+8000):
					_, data = wavfile.read(file_list[round(random.random()*1500)])
				out_source = data[8000:sig_len+8000]


			# t_source = dir_sig[0,0:sig_len]

			out_sig = np.zeros((conv_sig_len,hoa_dim),dtype='float32')
			for hoa_ii in range(hoa_dim):
				out_sig[:,hoa_ii] = np.convolve(out_source,hoa_tf[:,hoa_ii])
			out_sig = add_noise(out_sig,random.random()*25)
			ref_sig = np.dot(out_sig,hoa_coef.T)
			

			# mic_tf_t = add_noise(mic_tf,random.random()*15+25)
			# time_len2 = int(np.ceil(conv_sig_len*2/stft_len))+1
			# fft_mic_sig = np.zeros((freq_len,time_len2,mic_dim),dtype='complex_')
			# fft_ref_sig = np.zeros((freq_len,time_len2,ref_ch),dtype='complex_')
			# ref_sig = np.zeros((conv_sig_len,ref_ch),dtype='float32')
			# # mic_sig = np.zeros((conv_sig_len,mic_dim),dtype='float32')
			# for mic_ii in range(mic_dim):
			# 	mic_sig = np.convolve(out_source,mic_tf_t[:,mic_ii])
			# 	_,_,fft_dir = stft(mic_sig,fs=1,window='hann',nperseg = stft_len)
			# 	fft_mic_sig[:,:,mic_ii] = fft_dir
			# for freq_ii in range(1,freq_len):
			# 	temp = np.dot(TT[freq_ii-1,:,:],fft_mic_sig[freq_ii,:,:].T)
			# 	fft_ref_sig[freq_ii,:,:] = np.dot(temp.T,hoa_coef.T)
			# for ref_ii in range(ref_ch):
			# 	_,t_source = istft(fft_ref_sig[:,:,ref_ii],fs=1,window='hann',nperseg = stft_len)
			# 	ref_sig[:,ref_ii] = t_source[0:conv_sig_len]

			fft_align_sig = np.zeros((freq_len,time_len,ref_ch*2),dtype='float32')
			fft_dir_sig = np.zeros((freq_len,time_len,2),dtype='float32')
			ext_s = np.zeros((conv_sig_len),dtype='float32')
			ext_s[0:sig_len] = out_source
			st_ref_sig = np.zeros((sig_len,ref_ch),dtype='float32')

			_,_,fft_dir = stft(out_source,fs=1,window='hann',nperseg = stft_len)
			fft_dir_sig[:,:,0] = fft_dir.real
			fft_dir_sig[:,:,1] = fft_dir.imag

			for ch_ii in range(ref_ch):
				xcor = np.correlate(ext_s,ref_sig[:,ch_ii],'full')
				max_pos = np.argmax(xcor)
				if max_pos > conv_sig_len-1:
					sta_pos = max_pos-conv_sig_len+1
					st_ref_sig[sta_pos:sta_pos+sig_len,ch_ii] = ref_sig[0:sig_len-sta_pos,ch_ii]
				else:
					end_pos = conv_sig_len-max_pos-1
					st_ref_sig[0:sig_len-end_pos,ch_ii] = ref_sig[end_pos:sig_len,ch_ii]
				temp_sig = st_ref_sig[:,ch_ii]

				# xcor = np.correlate(ext_s,temp_sig,'full')
				# max_pos = np.argmax(xcor)
				# print(max_pos)	
			# for ch_ii in range(ref_ch):
			# 	temp_sig = ref_sig[delay_pot[ch_ii]:delay_pot[ch_ii]+sig_len,ch_ii]
				_,_,fft_ref = stft(temp_sig,fs=1,window='hann',nperseg = stft_len)
				fft_align_sig[:,:,ch_ii*2] = fft_ref.real
				fft_align_sig[:,:,ch_ii*2+1] = fft_ref.imag
				if ch_ii == 0:
					ext_s[0:sig_len] = st_ref_sig[:,0]
			
			fft_align_sig /= np.max(fft_align_sig)
			fft_dir_sig /= np.max(fft_dir_sig)
			batch_align_ref[batch_ii,:,:,:] = fft_align_sig
			batch_dir_sig[batch_ii,:,:,:] = fft_dir_sig


		group_ref_sig.append(batch_align_ref)
		group_dir_sig.append(batch_dir_sig)
				# if ch_ii>0:
				# 	max_pos[ch_ii-1] = np.argmax(np.correlate(align_sig[:,0],align_sig[:,ch_ii],'full'))
				# 	max_val[ch_ii-1] = np.max(np.correlate(align_sig[:,0],align_sig[:,ch_ii],'full'))
			# pdb.set_trace()
	return (group_ref_sig,group_dir_sig)
def creat_sig_for_pred(file_ii):

	data = loadmat(file_ii)
	# data = data["data"][0][0]
	# 
	ref_sig = data['data']['ref_sig'][0][0].astype(np.float32)
	dir_sig = data['data']['dir_sig1'][0][0].astype(np.float32)
	# hoa_tf = data['data']['hoa_tf'][0][0].astype(np.float32)
	# print(len(dir_sig))

	# print(ref_sig.shape)
	# print(dir_sig.shape)

	s_inf = data['data']['s_inf1'][0][0].astype(np.float32)
	ref_ch = len(ref_sig[:,1])
	
	sig_len = cfg.sig_len
	stft_len = cfg.stft_len
	time_len = int(np.ceil(sig_len*2/stft_len))+1
	freq_len = int(stft_len/2)+1
	batch_size = cfg.batch_size
	# out_ref_sig = np.zeros((len(ref_sig),sig_len,7),dtype='float32')
	# out_dir_sig = np.zeros((len(ref_sig),sig_len),dtype='float32')
	batch_align_ref = np.zeros((batch_size, freq_len,time_len,ref_ch*2), dtype=np.float32)
	batch_dir = np.zeros((batch_size, freq_len,time_len,2), dtype=np.float32)
	for batch_ii in range(batch_size):
		# batch_num = int(np.floor(random.random()*(len(dir_sig)-1)))
		# print(batch_num)
		fft_dir_sig = np.zeros((freq_len,time_len,2), dtype=np.float32)
		fft_align_sig = np.zeros((freq_len,time_len,2*ref_ch), dtype=np.float32)
		samp_dir_sig = dir_sig[0,0:sig_len]/np.max(dir_sig[0,0:sig_len])
		_,_,fft_dir = stft(samp_dir_sig,fs=1,window='hann',nperseg = stft_len)
		fft_dir_sig[:,:,0] = fft_dir.real
		fft_dir_sig[:,:,1] = fft_dir.imag
		batch_dir[batch_ii,:,:,:] = fft_dir_sig/np.max(fft_dir_sig)

		for ch_ii in range(ref_ch):
			# ref_sig[batch_num,ch_ii,:] = add_noise(ref_sig[batch_num,ch_ii,:],random.random()*15+15)
			_,_,fft_ref = stft(ref_sig[ch_ii,:],fs=1,window='hann',nperseg = stft_len)
			fft_align_sig[:,:,ch_ii*2] = fft_ref.real
			fft_align_sig[:,:,ch_ii*2+1] = fft_ref.imag

		fft_align_sig /= np.max(fft_align_sig)
		batch_align_ref[batch_ii,:,:,:] = fft_align_sig
	return batch_align_ref,s_inf,dir_sig


def PMSSL():

	gpu_list = [0]
	gpu_num=len(gpu_list)
	cfg.batch_size = 20


	SH_file = sys.path[0]+'/getSH_matrix.mat'
	data = loadmat(SH_file)
	SH_mat = data['getSH_mat']  # 360*180*25;azi_list:(-180:1:179)/180*pi; ele_list:(-90:1:89)/180*pi

	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
	max_epoch=50

	root_file_path = sys.path[0]+'/data/input/'
	sig_len = cfg.sig_len
	# for snr_ii in snr_list:
	# temp_file = root_file_path+'/t60-'+str(t60_ii)+'snr-'+str(snr_ii)+'/input/'
	file_list  =search_file(root_file_path)
	print(root_file_path)
	for file_ii in file_list:
		name=os.path.basename(file_ii)
		print(file_ii)

		with tf.Graph().as_default():
			loss_checker=MetricChecker(cfg)
			sess_config = tf.ConfigProto()
			sess_config.allow_soft_placement = True
			sess_config.gpu_options.allow_growth = True
			sess = tf.Session(config=sess_config)
			initializer = tf.random_normal_initializer(mean=cfg.init_mean,stddev=cfg.init_stddev)
			with tf.variable_scope("room_inference", initializer=None):
				in_folder=sys.path[0]+'/model/enhance-Unet-22-6-4'
				model_file=tf.train.latest_checkpoint(in_folder)
				print(model_file)
				# cfg.batch_size = 1
				model=enhance_net(sess=sess,config=cfg, num_gpu=gpu_num ,folder=in_folder, initializer=initializer)
				saver=tf.train.Saver(tf.global_variables())
				saver.restore(sess,model_file)
				loss_checker.reset_vir(sess)
				model.reset_loss(sess)
				train_loss = []
				dev_loss = []

			
				# data=data['data']
				for i_epoch in range(max_epoch):

				# pdb.set_trace()
				
				#  # with spatial information

					# train_batch = creat_train_sig_on_shape(file_ii,SH_mat,gpu_num)
					# # reverb_sig = creat_forward_model_sig(pred,hoa_tf,s_inf,SH_mat)
					# i_global,loss,pred=model.run_batch(train_batch)

				# with spatial and signal information

					[ref_sig,s_inf,dir_sig] = creat_sig_for_pred(file_ii)
					pred = model.get_pred(ref_sig)
					train_sig = creat_forward_model_sig(pred,s_inf,SH_mat,file_ii)
					# # pdb.set_trace()
					i_global,loss,pred=model.run_batch(train_sig)
					print('Epoch:',i_epoch,'Loss:',loss)


				# with signal information

				out_folder = sys.path[0]+'/data/PMSSL/'

				if not os.path.exists(out_folder):
					os.makedirs(out_folder)	


				pred = model.get_pred(ref_sig)
				stft_len = cfg.stft_len
				out_sig = pred[0][0][:,:,0]+1j*pred[0][0][:,:,1]
				_,time_sig = istft(out_sig,fs=1,window='hann',nperseg = stft_len)

				save_sig = time_sig[0:sig_len]


				out={'source':save_sig}
				out_name=os.path.join(out_folder,name)
				
				savemat(out_name,{'data':out})

def test_enhance_wrong_cond():

	gpu_list = [0]
	gpu_num=len(gpu_list)
	cfg.batch_size = 3


	SH_file = '/data/gs/room inference/calculate the RIR/matlab_doa_est/getSH_matrix.mat'
	data = loadmat(SH_file)
	SH_mat = data['getSH_mat']  # 360*180*25;azi_list:(-180:1:179)/180*pi; ele_list:(-90:1:89)/180*pi

	os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
	max_epoch=40

	snr_list = [-10,-5,0,5,10,15,20]
	t60 = 0.8
	root_file_path = '/data/gs/room inference/calculate the RIR/matlab_enhance_art/test_signal_for_enh/diff_snr_rev'+str(np.int(t60*1000))+'t6/'
	sig_len = cfg.sig_len
	temp_file = root_file_path+'/t60-'+str(t60)+'snr-20/input/'
	file_list  =search_file(temp_file)
	print(root_file_path)
	file_ii = file_list[0]
	# for file_ii in file_list:
	name=os.path.basename(file_ii)
	print(file_ii)

	with tf.Graph().as_default():
		loss_checker=MetricChecker(cfg)
		sess_config = tf.ConfigProto()
		sess_config.allow_soft_placement = True
		sess_config.gpu_options.allow_growth = True
		sess = tf.Session(config=sess_config)
		initializer = tf.random_normal_initializer(mean=cfg.init_mean,stddev=cfg.init_stddev)
		with tf.variable_scope("room_inference", initializer=None):
			in_folder='/data/gs/room inference/learn_room/model/enhance-Unet-22-6-4'
			model_file=tf.train.latest_checkpoint(in_folder)
			print(model_file)
			# cfg.batch_size = 1
			model=enhance_net(sess=sess,config=cfg, num_gpu=gpu_num ,folder=in_folder, initializer=initializer)
			saver=tf.train.Saver(tf.global_variables())
			saver.restore(sess,model_file)
			loss_checker.reset_vir(sess)
			model.reset_loss(sess)
			train_loss = []
			dev_loss = []

		
			# data=data['data']
			for i_epoch in range(max_epoch):

			# pdb.set_trace()
			
			#  # with spatial information

				# train_batch = creat_train_sig_on_shape(file_ii,SH_mat,gpu_num)
				# # reverb_sig = creat_forward_model_sig(pred,hoa_tf,s_inf,SH_mat)
				# i_global,loss,pred=model.run_batch(train_batch)

			# with spatial and signal information

				[ref_sig,s_inf,dir_sig] = creat_sig_for_pred(file_ii)
				pred = model.get_pred(ref_sig)
				train_sig = creat_forward_model_sig(pred,s_inf,SH_mat,file_ii)
				# # pdb.set_trace()
				i_global,loss,pred=model.run_batch(train_sig)


			# with signal information

			for file_ii in file_list:
				name=os.path.basename(file_ii)
				for snr_ii in snr_list:
					snr_file_path = root_file_path+'t60-'+str(t60)+'snr-'+str(snr_ii)
					input_folder = snr_file_path+'/input/'
					out_folder = snr_file_path+'/Unet-22-6-4-phy(psig+ideal-spa)-wrong/'

					if not os.path.exists(out_folder):
						os.makedirs(out_folder)	


					in_file_name = input_folder+name
					[ref_sig,s_inf,dir_sig] = creat_sig_for_pred(in_file_name)

					pred = model.get_pred(ref_sig)
					stft_len = cfg.stft_len
					out_sig = pred[0][0][:,:,0]+1j*pred[0][0][:,:,1]
					_,time_sig = istft(out_sig,fs=1,window='hann',nperseg = stft_len)

					save_sig = time_sig[0:sig_len]

					out_snr = si_snr(dir_sig,save_sig)
					print('in_snr:',snr_ii,'out_snr:',out_snr)

					out={'source':save_sig}
					out_name=os.path.join(out_folder,name)
					
					savemat(out_name,{'data':out})

def test_enhance_diff_cond():

	snr_list = [-5,0,5,10,15,20]
	root_file_path = '/data/gs/room inference/calculate the RIR/matlab_enhance_art/test_signal_for_enh/diff_snr_rev800t/'
	sig_len = cfg.sig_len

	for snr_ii in snr_list:
		snr_file_path = root_file_path+'t60-0.8snr-'+str(snr_ii)
		input_folder = snr_file_path+'/input/'
		out_folder = snr_file_path+'/Unet-22-6-4-phy/'

		if not os.path.exists(out_folder):
			os.makedirs(out_folder)	

		file_list  =search_file(input_folder)
		# pdb.set_trace()

		# gpu_list=cfg.gpu_list
		gpu_list = [0]
		gpu_num=len(gpu_list)
		cfg.batch_size = 3


		SH_file = '/data/gs/room inference/calculate the RIR/matlab_doa_est/getSH_matrix.mat'
		data = loadmat(SH_file)
		SH_mat = data['getSH_mat']  # 360*180*25;azi_list:(-180:1:179)/180*pi; ele_list:(-90:1:89)/180*pi

		os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
		max_epoch=15
		for file_ii in file_list:
			name=os.path.basename(file_ii)
			print(file_ii)
		# best_num=0
		# fig=plt.figure(1)
		# ax=fig.add_subplot(1,1,1)
		# plt.ion()
		# try:
			with tf.Graph().as_default():
				loss_checker=MetricChecker(cfg)
				sess_config = tf.ConfigProto()
				sess_config.allow_soft_placement = True
				sess_config.gpu_options.allow_growth = True
				sess = tf.Session(config=sess_config)
				initializer = tf.random_normal_initializer(mean=cfg.init_mean,stddev=cfg.init_stddev)
				with tf.variable_scope("room_inference", initializer=None):
					in_folder='/data/gs/room inference/learn_room/model/enhance-Unet-22-6-4'
					model_file=tf.train.latest_checkpoint(in_folder)
					print(model_file)
					# cfg.batch_size = 1
					model=enhance_net(sess=sess,config=cfg, num_gpu=gpu_num ,folder=in_folder, initializer=initializer)
					saver=tf.train.Saver(tf.global_variables())
					saver.restore(sess,model_file)
					loss_checker.reset_vir(sess)
					model.reset_loss(sess)
					train_loss = []
					dev_loss = []

				
					# data=data['data']
					for i_epoch in range(max_epoch):

					# pdb.set_trace()
					
					#  # with spatial information

						# train_batch = creat_train_sig_on_shape(file_ii,SH_mat,gpu_num)
						# # reverb_sig = creat_forward_model_sig(pred,hoa_tf,s_inf,SH_mat)
						# i_global,loss,pred=model.run_batch(train_batch)

					# with spatial and signal information

						[ref_sig,s_inf,dir_sig] = creat_sig_for_pred(file_ii)
						pred = model.get_pred(ref_sig)
						# print(pred[0][0].shape)
						train_sig = creat_forward_model_sig(pred,s_inf,SH_mat,file_ii)
						# # pdb.set_trace()
						i_global,loss,pred=model.run_batch(train_sig)



						# train_batch = creat_train_sig_on_shape(file_ii,SH_mat,gpu_num)
						# # reverb_sig = creat_forward_model_sig(pred,hoa_tf,s_inf,SH_mat)
						# i_global,loss,pred=model.run_batch(train_batch)

					# with signal information

						# [ref_sig,hoa_tf,s_inf,dir_sig] = creat_sig_for_pred(file_ii)

						# tf_file = '/data/gs/room inference/calculate the RIR/speech_signals/train_enhance_aligned_10r'
						# tf_file_list = search_file(tf_file)
						# # _, data = wavfile.read(file_list[round(random.random()*1500)])
						# [tref_sig,hoa_tf,s_inf,tdir_sig] = creat_sig_for_pred(tf_file_list[round(random.random()*1500)])

						# pred = model.get_pred(ref_sig)
						# reverb_sig = creat_forward_model_sig(pred,hoa_tf,s_inf,SH_mat,file_ii)
						# # # pdb.set_trace()
						# i_global,loss,pred=model.run_batch((reverb_sig,pred))

						# train_batch = creat_train_sig_on_shape(file_ii,SH_mat,gpu_num)
						# # reverb_sig = creat_forward_model_sig(pred,hoa_tf,s_inf,SH_mat)
						# i_global,loss,pred=model.run_batch(train_batch)

						pred = model.get_pred(ref_sig)
						stft_len = cfg.stft_len
						out_sig = pred[0][0][:,:,0]+1j*pred[0][0][:,:,1]
						_,time_sig = istft(out_sig,fs=1,window='hann',nperseg = stft_len)

						save_sig = time_sig[0:sig_len]

						out_snr = si_snr(dir_sig,save_sig)
						print('out_snr:',out_snr)

					out={'source':save_sig}
					out_name=os.path.join(out_folder,name)
					
					savemat(out_name,{'data':out})




		plt.ioff()
		# plt.ioff()
		# save_name = '/data/gs/room inference/calculate the RIR/matlab_doa_time_delay_enhance/enh_art/loss_data_sig_spa.m'
		# savemat(save_name,{'data':train_loss})
				# train_reader._producer.stop()
				# dev_reader._producer.stop()

def test_enhance_real():

	for f_ii in range(4,5):
		file_num = '%d'%(f_ii)
		folder='/data/gs/room inference/learn_room/model/enhance-6-6-phy-dist'+str(f_ii)
		if not os.path.exists(folder):
			os.makedirs(folder)	

		file_folder='/data/gs/room inference/calculate the RIR/speech_signals/test_doa_'+\
				'delay_enhance/3s_len_sig/r4_1_record(5)/dist'+str(f_ii)+'/input/'
		# file_folder='/data/gs/room inference/calculate the RIR/speech_signals/test_doa_'+\
		# 		'delay_enhance/3s_len_sig/diff_snr_1s/input/'

		file_list  =search_file(file_folder)
		stft_len = cfg.stft_len
		sig_len = cfg.sig_len
		time_len = int(np.ceil(sig_len*2/stft_len))+1
		freq_len = int(stft_len/2)+1
		# pdb.set_trace()

		# gpu_list=cfg.gpu_list
		gpu_list = [0]
		gpu_num=len(gpu_list)

		print(gpu_list)
		SH_file = '/data/gs/room inference/calculate the RIR/matlab_doa_est/getSH_matrix.mat'
		data = loadmat(SH_file)
		SH_mat = data['getSH_mat']  # 360*180*25;azi_list:(-180:1:179)/180*pi; ele_list:(-90:1:89)/180*pi

		os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))
		max_epoch=20
		best_num=0
		fig=plt.figure(f_ii)
		ax=fig.add_subplot(1,1,1)
		plt.ion()
		# try:
		with tf.Graph().as_default():
			loss_checker=MetricChecker(cfg)
			sess_config = tf.ConfigProto()
			sess_config.allow_soft_placement = True
			sess_config.gpu_options.allow_growth = True
			sess = tf.Session(config=sess_config)
			initializer = tf.random_normal_initializer(mean=cfg.init_mean,stddev=cfg.init_stddev)
			with tf.variable_scope("room_inference", initializer=None):
				in_folder='/data/gs/room inference/learn_room/model/enhance-Unet-22-6-6'
				model_file=tf.train.latest_checkpoint(in_folder)
				print(model_file)
				cfg.batch_size = 2
				model=enhance_net(sess=sess,config=cfg, num_gpu=gpu_num ,folder=folder, initializer=initializer)
				saver=tf.train.Saver(tf.global_variables())
				saver.restore(sess,model_file)
				loss_checker.reset_vir(sess)
				model.reset_loss(sess)
				train_loss = []
				dev_loss = []


				for i_epoch in range(max_epoch):
					logging.info("Start Epoch {}/{}".format(i_epoch + 1, max_epoch))
					while not loss_checker.should_stop():
						# pdb.set_trace()
						for file_ii in file_list:

						# with spatial and signal information

							[ref_sig,hoa_tf,s_inf,dir_sig] = creat_sig_for_pred(file_ii)
							pred = model.get_pred(ref_sig)
							reverb_sig = creat_forward_model_sig(pred,hoa_tf,s_inf,SH_mat,file_ii)
							# # # pdb.set_trace()
							i_global,loss,pred=model.run_batch((reverb_sig,pred))

							train_batch = creat_train_sig_on_shape(file_ii,SH_mat,gpu_num)
							i_global,loss,pred=model.run_batch(train_batch)

							group_ref_sig = []
							group_dir_sig = []


							# for gpu_ii in range(gpu_num):

							# 	batch_align_ref = np.zeros((cfg.batch_size, freq_len,time_len,14), dtype=np.float32)
							# 	batch_dir = np.zeros((cfg.batch_size, freq_len,time_len,2), dtype=np.float32)
							# 	batch_align_ref[0:int(cfg.batch_size/2),:,:,:] = reverb_sig[gpu_ii][0:int(cfg.batch_size/2),:,:,:]
							# 	batch_align_ref[int(cfg.batch_size/2):,:,:,:] = train_batch[0][gpu_ii]
							# 	batch_dir[0:int(cfg.batch_size/2),:,:,:] = pred[gpu_ii][0:int(cfg.batch_size/2),:,:,:]
							# 	batch_dir[int(cfg.batch_size/2):,:,:,:] = train_batch[1][gpu_ii]
							# 	group_ref_sig.append(batch_align_ref)
							# 	group_dir_sig.append(batch_dir)

							# # reverb_sig = creat_forward_model_sig(pred,hoa_tf,s_inf,SH_mat)
							# i_global,loss,pred=model.run_batch((group_ref_sig,group_dir_sig))


							print('i_global:',i_global,'loss:',loss[0])
							if i_global % 1 == 0:

								train_loss.append(loss[0])
								# dev_loss.append(avg_loss)
								try:
									ax.lines.remove(lines1[0])
									# ax.lines.remove(lines2[0])
								except  Exception as e:
									print(e)
								lines1 = ax.plot(train_loss,label = "train loss")
								# lines2 = ax.plot(dev_loss,label = "dev loss")
								plt.pause(0.1)
								loss_improved, best_loss = loss_checker.update(sess,loss[0])
								if loss_improved:
									# logging.info("new best loss {}".format(best_loss))
									print("new best loss:",best_loss)
									model_path=os.path.join(folder,'model.ckpt')
									save=tf.train.Saver(tf.global_variables())
									save.save(sess,model_path)
									best_num=best_num+1  
						break 
					if loss_checker.should_stop():
						logging.info("Early stopped")
						break
		plt.ioff()
		save_name = '/data/gs/room inference/calculate the RIR/matlab_doa_time_delay_enhance/enh_art/loss_data.m'
		savemat(save_name,{'data':train_loss})
				# train_reader._producer.stop()
				# dev_reader._producer.stop()


def tst():
	room_shape = np.array([6.37,4.68,2.34])
	mic_pos = room_shape - np.array([1.7,1.7,1.39])
	source_pos = room_shape - np.array([2.61,2.19,1.39])
	SH_file = '/data/gs/room inference/calculate the RIR/matlab_doa_est/getSH_matrix.mat'
	data = loadmat(SH_file)
	SH_mat = data['getSH_mat']  # 360*180*25;azi_list:(-180:1:179)/180*pi; ele_list:(-90:1:89)/180*pi
	# hoa_tf,s_inf = creat_reverb_tf(SH_mat,mic_pos,source_pos,room_shape)
	# hoa_tf = add_noise(hoa_tf,5)
	# print(s_inf)
	# plt.figure()
	# plt.plot(hoa_tf[:,1])
	# plt.show()
	# pdb.set_trace()
	print(np.random.rand(5))
def check_hoa_tf():
	file_ii = '/data/gs/room inference/calculate the RIR/matlab_enhance_art/test_signal_for_enh/diff_snr_rev800/t60-0.8snr-20/input/1-1.mat'
	data = loadmat(file_ii)
	data = data["data"][0][0]

	# TT = data['TT']
	# error_ind = cfg.error_ind
	# 
	room_shape = data['room_shape'][0]
	print(room_shape)
	mic_pos = data['mic_pos'][0]
	source_pos = data['source_pos'][0]
	# mic_pos = room_shape-mic_pos
	# source_pos  =room_shape-source_pos
	mic_dim = 32
	SH_file = '/data/gs/room inference/calculate the RIR/matlab_doa_est/getSH_matrix.mat'
	data = loadmat(SH_file)
	SH_mat = data['getSH_mat']  # 360*180*25;azi_list:(-180:1:179)/180*pi; ele_list:(-90:1:89)/180*pi

	# room_shape = room_shape+(np.random.rand(3)-0.5)
	# mic_pos = mic_pos+(np.random.rand(3)-0.5)/5
	# source_pos = source_pos+(np.random.rand(3)-0.5)/5

	hoa_tf,s_inf = creat_reverb_tf(SH_mat,mic_pos,source_pos,room_shape)
	plt.figure(1)
	plt.plot(hoa_tf[:,10])
	plt.show()



	# print(sample_rate)
	# print(len(data)/sample_rate)

if __name__ == "__main__":
	# test_enhance_real()
	# test_enhance()
	# check_hoa_tf()
	PMSSL()
	# tst()


	
