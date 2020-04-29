# -*- coding: utf-8 -*-

#basic constants
N_EPOCHS = 1
CLIP = 0.1
BATCH_SIZE = 100
N_FOLDS = 5
SEED = 1234
LR = 0.01
WEIGHT_DECAY = 0.01


#CNN constants
CNN_EPOCHS = 15
EMB_DIM = 300
HID_DIM = 512 # each conv. layer has 2 * hid_dim filters
ENC_LAYERS = 10 # number of conv. blocks in encoder
DEC_LAYERS = 10 # number of conv. blocks in decoder
ENC_KERNEL_SIZE = 3 # must be odd!
DEC_KERNEL_SIZE = 3 # can be even or odd
CNN_ENC_DROPOUT = 0.25
CNN_DEC_DROPOUT = 0.25


#RNN with Attention constants
ATTN_EPOCHS = 15
LAYERS  = 2
ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ATTN_ENC_DROPOUT = 0.5
ATTN_DEC_DROPOUT = 0.5

#COPYNET constants
COPYNET_EPOCHS = 15
