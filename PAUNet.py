import sys
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models, losses
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.optimizers import Optimizer
#----------------------------------------------------------------
#Define all the customised functions
#----------------------------------------------------------------
# Non-linear Activation switch - 0: linear; 1: non-linear
# If non-linear, then Leaky ReLU Activation function with alpha value as input
def actv_swtch(swtch, alpha_val):
    if swtch == 0:
        actv = "linear"
    else:
        actv = layers.LeakyReLU(alpha=alpha_val)
    return actv
def con2d(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):
    return layers.Conv2D(n_out, (fil, fil), dilation_rate=(dil_rate, dil_rate),
                                  strides=std,
                                  activation=actv_swtch(swtch, alpha_val),
                                  padding="same",
                                  use_bias=True,
                                  kernel_regularizer=l1(reg_val),
                                  bias_regularizer=l1(reg_val))(inp)

#defining a generic 2D transposed convolution layer for our use
#Inputs:same as con2d
def con2d_trans(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):
    return layers.Conv2DTranspose(n_out,(fil, fil),dilation_rate=(dil_rate,dil_rate),
                         strides=std,
                         activation=actv_swtch(swtch, alpha_val),
                         padding="same",
                         use_bias=True,
                         kernel_regularizer=l1(reg_val),
                         bias_regularizer=l1(reg_val))(inp)

#Center crop the map/image
def center_crop(inp):
    start = math.floor(inp.shape[1] / 4)
    end = start + math.ceil(inp.shape[1] / 2)
    cen_crop = inp[:, start:end, start:end, :]
    return cen_crop

def Attention(inp,num_heads, key_dim, attention_axes):
        layer = layers.MultiHeadAttention(num_heads, key_dim, attention_axes=None)
        y = layer(inp, inp)
        return y

#Convolution encoder layers stack
def encoder_layer(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):
    y = con2d(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = y+con2d(inp, n_out, 1, dil_rate, std, swtch, alpha_val, reg_val)
    y_crop = center_crop(y)
    return y_crop, y 

#Convolution decoder layers stack
def decoder_layer(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):
    y = con2d_trans(inp, n_out, fil, dil_rate, std*2, swtch, alpha_val, reg_val)
    inp_ = y
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = y + con2d(inp_, n_out, 1, dil_rate, std, swtch, alpha_val, reg_val)
    return y  


def PAUNet(n_blocks, MHA_blocks, lat, lon, chnl, out_days, n_out, fil, dil_rate, 
          std, swtch, alpha_val, reg_val, num_heads, key_dim, attention_axes):
    inp  = layers.Input(shape = (lat, lon, chnl))
    y = inp
    # --- Encoder route ---#
    for n_block in range(n_blocks):
        if n_block == n_blocks-1:
            _,y = encoder_layer(y, n_out, fil, dil_rate,std, swtch, alpha_val, reg_val)
        else:
            y,_ = encoder_layer(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
        n_out = n_out * 2
    #Concat center crop of satilite radiance maps to encoder output before attention
    y = layers.Concatenate(axis=-1)([y, center_crop(center_crop(inp))])
    #Multihead attention blocks
    for MHA_block in range(MHA_blocks):
        y = Attention(y,num_heads, key_dim, attention_axes)
    # --- Decoder route ---#
    n_out = n_out // 2
    for n_block in range(n_blocks-2,-1,-1):
        n_out = n_out // 2
        y = decoder_layer(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y_hat = con2d(y, out_days, fil, dil_rate, std, swtch, alpha_val, reg_val)
    return models.Model(inputs = inp, outputs = y_hat)

'''
args: 
n_blocks: number of convolution blocks
MHA_blocks: number of multihead attention blocks
lat: latitude dimension
lon: longitude dimension
chnl: number of input channels
out_days: length of prediction sequence times
n_out: number of feature maps
fil: filter size
dil_rate: dialation rate
std: stride
swtch: activation switch, 0 for linear and 1 for nonlinear
alpha_val: 0 for relu and alpha for leaky relu
reg_val: L1 regularizer value
num_heads: Attention heads
key_dim: Attention head size
attention_axes: Attention axes}
'''
n_seq_predict = np.int(sys.argv[1])
model = PAUNet(3, 1, 252, 252, 44, n_seq_predict, 64, 3, 1, 1, 1, 0, 1e-5, 1, 64, None)
print(model.summary())
