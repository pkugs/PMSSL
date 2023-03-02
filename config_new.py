# config file for training or other process
# check the config file before your process
# and import this file to get parameter

# data config
# samp_rate = 16000
# frame_duration = 0.032
# frame_size = int(samp_rate * frame_duration)
hoa_dim = 25
ref_ch = 7
gamma_num = 32
# dir_ch = 24
sig_len = int(16000*3)
speech_len=40  # the room estimation length of speech frame
shape_len=7200			
pos_len=10000
ang_ch=24
# shape_len = 154
send_len=1
fft_len = sig_len
fs = 16000
one_hot_len = 1000
stft_len = 1024
order = 4
batch_size = 5
dev_batch_size = 5
min_queue_size = 4
load_file_num = 5
# min_sent_len = 10


train_ratio=0.8
dev_ratio=0.1
test_ratio=0.1

# data_path

# job config

gpu_list = [0,1]

# model config
log_feat = True
global_cmvn_file = ''
# t12
# k_s = 20
# conv_channels = [40,40]
# conv_kernels = [k_s,k_s,k_s,k_s]  # each item stand for the kernel for eavh conv layer
#                            # kernel shape is [time, frequency]
# conv_stride = [[1],[1],[1],[1]]    # strides for batch and channel are 1 by default
# conv_dilation = [1,2,4]
# conv1_channels = [30,50,100,60]
# conv1_kernels=[k_s,k_s,k_s,k_s,k_s,k_s,k_s,k_s,k_s] 
# conv1_stride = [[1],[4],[8],[8],[5],[4],[3],[3]]
# conv2_channels = [30,50,100,60]
# conv2_kernels=[k_s,k_s,k_s,k_s,k_s,k_s,k_s,k_s,k_s] 
# conv2_stride = [[1],[4],[8],[8],[5],[4],[3],[3]]
# keep_prob=0.9
# t3
k_s = 30
conv3_channels = [50,150,24]
conv3_kernels=[k_s,k_s,k_s,k_s,k_s,k_s,k_s,k_s,k_s] 
conv3_stride = [5,2,1]
keep_prob=0.9

num_rnn_layers = 1
num_rnn_layers_t1=1
num_rnn_layers_t2=1
num_rnn_layers_t3=1
rnn_type = 'lstm'  # support 'lstm', 'gru'
hidden_size=[30]
hidden_size_t1=[60,60,120]
hidden_size_t2=[30,30,40]
hidden_size_t3=[10,20]
convlstm1_filters = 20
convlstm1_kernel = [5,5]
convlstm2_filters = 20
convlstm2_kernel = [5,5]
convlstm3_filters = 20
convlstm3_kernel = [5,5]

bidirectional = True
pred_mask = True

# training param
seed = 123
resume = False
init_mean = 0.0
init_stddev = 0.02
max_grad_norm = 100
learning_rate = 1e-3
max_epoch = 50
s_max_epoch = 3
pretrain_shorter_epoch = 10
log_period = 15
# save_period = 70
dev_period = 100
early_stop_count = 200
decay_lr_count = 3
decay_lr = 0.5
min_learning_rate = 1e-6

# test_config
test_noisy_list = "./list/ch25/dev_noise.lst"
test_clean_list = "./list/ch25/dev_clean.lst"
test_fading = True
test_name = "job/HOA_NN_ch25/testOndev/"

# load option 
load_option = 1
load_path = ''
