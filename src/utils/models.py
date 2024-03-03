# keras 2.2.4

import keras
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
#from keras.layers.convolutional_recurrent import ConvLSTM2D
from utils.convolutional_recurrent_test import ConvLSTM2D
from keras.layers import Input, MaxPooling2D, TimeDistributed, UpSampling2D, concatenate, subtract, add, BatchNormalization, Lambda, multiply
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model
from keras import backend as K
# from keras_contrib.losses import DSSIMObjective

# model

def res_layer(inputs,timesteps):
    out = []
    for i in range(timesteps-1):
        res = subtract([Lambda(lambda x: x[:,i+1,:,:,:])(inputs) , Lambda(lambda x: x[:,i,:,:,:])(inputs)])
        out.append(Lambda(lambda x: K.expand_dims(x, axis=1))(res))
    
    out = concatenate(out, axis=1)
    return out


def convLSTM_base(timesteps=None, input_width=224, input_height=224, input_channel=1,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error' ) :
    
    #input layer
    img_input = Input(shape=(None, input_width, input_height, 1))

    # unet block1
    conv1 = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', 
                       name='block1_conv1', return_state=False, return_sequences=True)(img_input)
    conv1 = BatchNormalization()(conv1)

    conv1 = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', 
                       name='block1_conv2', return_state=False, return_sequences=True)(conv1)
    conv1 = BatchNormalization()(conv1)

    conv1 = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', 
                       name='block1_conv3', return_state=False, return_sequences=True)(conv1)
    conv1 = BatchNormalization()(conv1)

    conv1 = ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', 
                       name='block1_conv4', return_state=False, return_sequences=True)(conv1)
    conv1 = BatchNormalization()(conv1)

    out = Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same', name='out')(conv1)

    model= Model(input = img_input, output = out)
    
    plot_model(model, to_file='./model_png/convLSTM_base.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
        
    model.compile(optimizer = optimizer, loss = loss)
    
    return model

def convLSTM_auto_base(timesteps=None, input_width=224, input_height=224, input_channel=4,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error',
                  lr = 0.001, decay = 0.00001) :
    

    encoder_inputs  = Input(shape=(timesteps, None, None, input_channel))
    gray_e = Lambda(lambda x: x[:,:,:,:,:1])(encoder_inputs)
    
    encoder_outputs, state_h, state_c = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same' 
                           , return_state=True, return_sequences=True)(gray_e)


    decoder_inputs = Input(shape=(timesteps, None, None, input_channel))
    gray_d = Lambda(lambda x: x[:,:,:,:,:1])(decoder_inputs)
    
    decoder_outputs,_,_ = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same'
                              , return_state=True, return_sequences=True)([gray_d, state_h, state_c])
    decoder_outputs = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same'
                              , return_state=False, return_sequences=True)(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    plot_model(model, to_file='./model_png/convLSTM_auto_base.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    
    if optimizer =='adam':
        opt = keras.optimizers.Adam(lr=lr, decay=decay)
        
    model.compile(optimizer = opt, loss = loss)
    
    return model

def convLSTM_auto_each_base(timesteps=None, input_width=224, input_height=224, input_channel=4,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error',
                  lr = 0.001, decay = 0.00001) :
    

    encoder_inputs  = Input(shape=(timesteps, input_width, input_height, input_channel))
    encoder_outputs, state_h, state_c = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same' 
                           , return_state=True, return_sequences=True)(encoder_inputs)


    decoder_inputs = Input(shape=(timesteps, input_width, input_height, input_channel))
    decoder_outputs,_,_= ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'
                              , return_state=True, return_sequences=True)([decoder_inputs, state_h, state_c])
    out = []
    for i in range(timesteps):
        de_each = Lambda(lambda x: x[:,i,:,:,:])(decoder_outputs)
        conv_each = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(de_each)
        out.append(Lambda(lambda x: K.expand_dims(x, axis=1))(conv_each))
    out = concatenate(out, axis=1)
    
    model = Model([encoder_inputs, decoder_inputs], out)

    plot_model(model, to_file='./model_png/convLSTM_auto_base2.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
        
    if optimizer =='adam':
        opt = keras.optimizers.Adam(lr=lr, decay=decay)
        
    model.compile(optimizer = opt, loss = loss)
    
    return model

def convLSTM_auto_new(timesteps=None, input_width=224, input_height=224, input_channel=1,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error' ) :
    

    encoder_input  = Input(shape=(timesteps, input_width, input_height, input_channel))
    encoder_inputs = TimeDistributed(Conv2D(filters=32,  kernel_size=(3, 3), padding='same'))(encoder_input)
    encoder_inputs = TimeDistributed(Conv2D(filters=32,  kernel_size=(3, 3), padding='same'))(encoder_inputs)
    encoder_inputs = TimeDistributed(MaxPooling2D((2,2)))(encoder_inputs)

    encoder_outputs, state_h, state_c = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same' 
                           , return_state=True, return_sequences=True)(encoder_inputs)


    decoder_input  = Input(shape=(timesteps, input_width, input_height,input_channel))
    decoder_inputs = TimeDistributed(Conv2D(filters=32,  kernel_size=(3, 3), padding='same'))(decoder_input)
    decoder_inputs = TimeDistributed(Conv2D(filters=32,  kernel_size=(3, 3), padding='same'))(decoder_inputs)
    decoder_inputs = TimeDistributed(MaxPooling2D((2,2)))(decoder_inputs)

    decoder_outputs = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'
                              , return_state=False, return_sequences=True)([decoder_inputs, state_h, state_c])

    decoder_outputs = TimeDistributed(UpSampling2D((2, 2)))(decoder_outputs)
    decoder_outputs = TimeDistributed(Conv2D(filters=32,  kernel_size=(3, 3), padding='same'))(decoder_outputs)
    decoder_outputs = TimeDistributed(Conv2D(filters=32,  kernel_size=(3, 3), padding='same'))(decoder_outputs)

    decoder_outputs = TimeDistributed(Conv2D(filters=input_channel,  kernel_size=(3, 3), padding='same'))(decoder_outputs)
    
    model = Model([encoder_input,decoder_input], decoder_outputs)
    plot_model(model, to_file='./model_png/convLSTM_auto_new.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
        
    model.compile(optimizer = optimizer, loss = loss)
    
    return model

def convLSTM(timesteps=None, input_width=224, input_height=224, input_channel=1,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error' ) :
    
    #input layer
    img_input = Input(shape=(timesteps, input_width, input_height, input_channel))
    
    # unet block1
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(img_input)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(conv1)

    pool1 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(conv1)

    # unet block2
    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(pool1)

    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(conv2)

    # upsampling1
    up1 = TimeDistributed(UpSampling2D((2, 2)))(conv2)
    up1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(up1)
    up1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(up1)
    up1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                       return_state=False, return_sequences=True)(up1)
    out = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', 
                        return_state=False, return_sequences=True)(up1)

    model= Model(input = img_input, output = out)
    
    plot_model(model, to_file='./model_png/convLSTM.png', show_shapes=True, show_layer_names=True)
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    print(model.summary())

    model.compile(optimizer = optimizer, loss = loss)
    
    return model

def convLSTM_unet(timesteps=None, input_width=224, input_height=224, input_channel=1,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error' ) :
    
    #input layer
    img_input = Input(shape=(timesteps, input_width, input_height, input_channel))
    
    # unet block1
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(img_input)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(conv1)

    pool1 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(conv1)

    # unet block2
    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(pool1)

    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(conv2)

    # upsampling1
    up1 = TimeDistributed(UpSampling2D((2, 2)))(conv2)
    up1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(up1)
    merge1 = concatenate([conv1,up1])
    up1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(merge1)
    up1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                       return_state=False, return_sequences=True)(up1)
    out = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', 
                        return_state=False, return_sequences=True)(up1)

    model= Model(input = img_input, output = out)
    
    plot_model(model, to_file='./model_png/convLSTM_unet.png', show_shapes=True, show_layer_names=True)
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    print(model.summary())

    model.compile(optimizer = optimizer, loss = loss)
    
    return model


def convLSTM_each_unet(timesteps=None, input_width=224, input_height=224, input_channel=1,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error' ) :
    #input layer
    img_input = Input(shape=(timesteps, None, None, input_channel))

    # unet block1
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(img_input)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', 
                        return_state=False, return_sequences=True)(conv1)
    conv1_each = []
    for i in range(timesteps) :
        conv1_each.append(Lambda(lambda x: x[:,i,:,:,:])(conv1))

    pool1 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(conv1)

    # unet block2
    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', 
                       return_state=False, return_sequences=True)(pool1)

    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', 
                       return_state=False, return_sequences=True)(conv2)

    # upsampling1
    up = TimeDistributed(UpSampling2D((2, 2)))(conv2)

    out = []
    for i in range(timesteps) :
        up_each = Lambda(lambda x: x[:,i,:,:,:])(up)
        up_conv = Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same')(up_each)
        concat = concatenate([conv1_each[i],up_conv])

        up_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(concat)
        up_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(up_conv)
        out_each =  Conv2D(filters=1, kernel_size=(3, 3), padding='same')(up_conv)

        out.append(Lambda(lambda x: K.expand_dims(x, axis=1))(out_each))

    out = concatenate(out, axis=1)    

    model= Model(input = img_input, output = out)
    
    plot_model(model, to_file='./model_png/convLSTM_each_unet.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
        
    model.compile(optimizer = optimizer, loss = loss)
    
    return model

def convLSTM_each_unet_res(timesteps=None, input_width=224, input_height=224, input_channel=1,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error' ) :
    
    img_input = Input(shape=(timesteps, None, None, input_channel))
    res = res_layer(img_input, 3)

    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(res)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(conv1)
    conv1_each = []
    for i in range(timesteps-1) :
        conv1_each.append(Lambda(lambda x: x[:,i,:,:,:])(conv1))

    pool1 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(conv1)

    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(pool1)
    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(conv2)

    up = TimeDistributed(UpSampling2D((2, 2)))(conv2)

    out = []
    for i in range(timesteps-1) :
        up_each = Lambda(lambda x: x[:,i,:,:,:])(up)
        up_conv = Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same')(up_each)
        concat = concatenate([conv1_each[i],up_conv])

        up_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(concat)
        up_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(up_conv)
        out_each =  Conv2D(filters=1, kernel_size=(1, 1), padding='same')(up_conv)

        out.append(Lambda(lambda x: K.expand_dims(x, axis=1))(out_each))

    out = concatenate(out, axis=1)    
    added = add([Lambda(lambda x: x[:,:2,:,:,:])(img_input),out])
    model= Model(inputs = img_input, outputs = added)      
    
    plot_model(model, to_file='./model_png/convLSTM_each_unet_res.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    #optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0001)  
    model.compile(optimizer = optimizer, loss = loss)
    
    return model


def convLSTM_each_unet_res_lite(timesteps=None, input_width=224, input_height=224, input_channel=1,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error' ) :
    
    img_input = Input(shape=(timesteps, None, None, input_channel))
    res = res_layer(img_input, 3)

    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(res)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(conv1)
    conv1_each = []
    for i in range(timesteps-1) :
        conv1_each.append(Lambda(lambda x: x[:,i,:,:,:])(conv1))

    pool1 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(conv1)

    conv2 = ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(pool1)
    conv2 = ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(conv2)

    up = TimeDistributed(UpSampling2D((2, 2)))(conv2)

    out = []
    for i in range(timesteps-1) :
        up_each = Lambda(lambda x: x[:,i,:,:,:])(up)
        up_conv = Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same')(up_each)
        concat = concatenate([conv1_each[i],up_conv])

        up_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(concat)
        up_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(up_conv)
        up_conv =  Conv2D(filters=2, kernel_size=(3, 3), activation='relu', padding='same')(up_conv)
        out_each =  Conv2D(filters=1, kernel_size=(1, 1), padding='same')(up_conv)

        out.append(Lambda(lambda x: K.expand_dims(x, axis=1))(out_each))

    out = concatenate(out, axis=1)    
    added = add([Lambda(lambda x: x[:,:2,:,:,:])(img_input),out])
    model= Model(inputs = img_input, outputs = added)      
    
    plot_model(model, to_file='./model_png/convLSTM_each_unet_res_lite.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    #optimizer = keras.optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0001)  
    model.compile(optimizer = optimizer, loss = loss)
    
    return model


def convLSTM_each_unet_res_new(timesteps=None, input_width=224, input_height=224, input_channel=1,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error' ) :
    
    img_input = Input(shape=(timesteps, None, None, input_channel))

    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(img_input)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(conv1)
    
    conv1_res = res_layer(conv1, 3)
    conv1_res_each = []
    for i in range(timesteps-1) :
        conv1_res_each.append(Lambda(lambda x: x[:,i,:,:,:])(conv1_res))

    pool1 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(conv1)

    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(pool1)
    conv2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True)(conv2)

    up = TimeDistributed(UpSampling2D((2, 2)))(conv2)
    res = res_layer(up, 3)
    
    out = []
    for i in range(timesteps-1) :
        up_each = Lambda(lambda x: x[:,i,:,:,:])(res)
        up_conv = Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same')(up_each)
        concat = concatenate([conv1_res_each[i],up_conv])

        up_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(concat)
        up_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(up_conv)
        out_each =  Conv2D(filters=1, kernel_size=(1, 1), padding='same')(up_conv)

        out.append(Lambda(lambda x: K.expand_dims(x, axis=1))(out_each))

    out = concatenate(out, axis=1)    
    added = add([Lambda(lambda x: x[:,:2,:,:,:])(img_input),out])
    model= Model(inputs = img_input, outputs = added)      
    
    plot_model(model, to_file='./model_png/convLSTM_each_unet_res_new.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0001)  
    model.compile(optimizer = optimizer, loss = loss)
    
    return model

def unet_res(timesteps=None, input_width=224, input_height=224, input_channel=1,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error' ) :
    
    img_input = Input(shape=(timesteps, None, None, input_channel))
    res = res_layer(img_input, 3)

    conv_each = []
    for i in range(timesteps - 1):
        res_each = Lambda(lambda x: x[:,i,:,:,:])(res)
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(res_each)
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(pool1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(conv2)
        
        up1 = Conv2D(filters=32, kernel_size=(2, 2), kernel_initializer='he_normal', activation='relu', padding='same')(UpSampling2D((2, 2))(conv2))
        concat = concatenate([conv1,up1]) 
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal',activation='relu', padding='same')(concat)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(conv3)
        conv3 = Conv2D(filters=2, kernel_size=(3, 3), kernel_initializer='he_normal', activation='relu', padding='same')(conv3)
        conv3 = Conv2D(filters=1, kernel_size=(3, 3),padding='same')(conv3)
        conv_each.append(Lambda(lambda x: K.expand_dims(x, axis=1))(conv3))
        
    out = concatenate(conv_each, axis=1)     
    added = add([Lambda(lambda x: x[:,:2,:,:,:])(img_input),out])
    model= Model(inputs = img_input, outputs = added)      
    
    plot_model(model, to_file='./model_png/unet_res.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    #optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0001)  
    model.compile(optimizer = optimizer, loss = loss)
    
    return model


def convLSTM_res(timesteps=None, input_width=224, input_height=224, input_channel=4,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error',
                  lr = 0.001, decay = 0.00001) :
    
    #input layer
    img_input = Input(shape=(timesteps, None, None, input_channel))
    flow = Lambda(lambda x: x[:,1:,:,:,1:])(img_input)
    
    # unet block1
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(flow)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv1)
    conv1 = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True)(conv1)
    add1 = add([Lambda(lambda x: x[:,:2,:,:,:1])(img_input),conv1])

    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(add1)
    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    concat2 = concatenate([add1,conv2])
    conv2 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat2)
    
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv3)
    concat3 = concatenate([conv2,conv3])
    conv3 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat3)
    
    conv4 = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True)(conv3)
    
    added = add([Lambda(lambda x: x[:,:2,:,:,:1])(img_input),conv4])
    
    model= Model(input = img_input, output = added)
    
    plot_model(model, to_file='./model_png/convLSTM_res.png', show_shapes=True, show_layer_names=True)
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    print(model.summary())
    
    if optimizer =='adam':
        opt = keras.optimizers.Adam(lr=lr, decay=decay)
        
    model.compile(optimizer = opt, loss = loss)
    return model
    

def convLSTM_res_test(timesteps=None, input_width=224, input_height=224, input_channel=4,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error',
                  lr = 0.001, decay = 0.00001) :
    
    #input layer
    img_input = Input(shape=(timesteps, None, None, input_channel))
    flow = Lambda(lambda x: x[:,1:,:,:,1:])(img_input)
    
    # unet block1
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(flow)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv1)
    conv1 = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True)(conv1)
    add1 = add([Lambda(lambda x: x[:,:2,:,:,:1])(img_input),conv1], name = 'loss1')

    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(add1)
    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    concat2 = concatenate([add1,conv2])
    conv2 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat2)
    
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv3)
    concat3 = concatenate([conv2,conv3])
    conv3 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat3)
    
    conv4 = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True)(conv3)
    
    added = add([add1,conv4], name = 'loss2')
    
    model= Model(input = img_input, output = [add1, added])
    
    plot_model(model, to_file='./model_png/convLSTM_res_test.png', show_shapes=True, show_layer_names=True)
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    print(model.summary())
    
    if optimizer =='adam':
        opt = keras.optimizers.Adam(lr=lr, decay=decay)
        
    model.compile(optimizer = opt, loss = {'loss1' : loss, 'loss2' : loss}, loss_weights = {'loss1' : 1, 'loss2' : 1})
    return model

def convLSTM_res_attention(timesteps=None, input_width=224, input_height=224, input_channel=4,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error',
                  lr = 0.001, decay = 0.00001) :
    
    #input layer
    img_input = Input(shape=(timesteps, None, None, input_channel))
    flow = Lambda(lambda x: x[:,1:,:,:,1:])(img_input)
    
    # unet block1
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(flow)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv1)
    conv1 = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True)(conv1)
    add1 = multiply([Lambda(lambda x: x[:,:timesteps-1,:,:,:1])(img_input),conv1])

    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(add1)
    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    concat2 = concatenate([add1,conv2])
    conv2 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat2)
    
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv3)
    concat3 = concatenate([conv2,conv3])
    conv3 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat3)
    
    conv4 = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True)(conv3)
    
    added = add([Lambda(lambda x: x[:,:timesteps-1,:,:,:1])(img_input),conv4])
    
    model= Model(input = img_input, output = added)
    
    plot_model(model, to_file='./model_png/convLSTM_res_attention.png', show_shapes=True, show_layer_names=True)
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    print(model.summary())
    
    if optimizer =='adam':
        opt = keras.optimizers.Adam(lr=lr, decay=decay)
        
    model.compile(optimizer = opt, loss = loss)
    return model

def convLSTM_res_attention_pred(timesteps=None, input_width=224, input_height=224, input_channel=4,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error',
                  lr = 0.001, decay = 0.00001) :
    
    #input layer
    img_input = Input(shape=(timesteps, None, None, input_channel))
    flow = Lambda(lambda x: x[:,1:,:,:,1:])(img_input)
    
    # unet block1
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(flow)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv1)
    conv1 = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True)(conv1)
    add1 = multiply([Lambda(lambda x: x[:,1:,:,:,:1])(img_input),conv1])

    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(add1)
    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    concat2 = concatenate([add1,conv2])
    conv2 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat2)
    
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv3)
    concat3 = concatenate([conv2,conv3])
    conv3 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat3)
    
    conv4 = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=False)(conv3)
    
    added = add([Lambda(lambda x: x[:,-1,:,:,:1])(img_input),conv4])
    
    model= Model(input = img_input, output = added)
    
    #plot_model(model, to_file='./model_png/convLSTM_res_attention.png', show_shapes=True, show_layer_names=True)
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    print(model.summary())
    
    if optimizer =='adam':
        opt = keras.optimizers.Adam(lr=lr, decay=decay)
        
    model.compile(optimizer = opt, loss = loss)
    return model

def convLSTM_res_pred(timesteps=None, input_width=224, input_height=224, input_channel=4,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error',
                  lr = 0.001, decay = 0.00001) :
    
    #input layer
    img_input = Input(shape=(timesteps, None, None, input_channel))
    flow = Lambda(lambda x: x[:,1:,:,:,1:])(img_input)
    
    # unet block1
  
    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(flow)
    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    concat2 = concatenate([flow,conv2])
    conv2 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat2)
    
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv3)
    concat3 = concatenate([conv2,conv3])
    conv3 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat3)
    
    conv4 = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=False)(conv3)
    
    added = add([Lambda(lambda x: x[:,-1,:,:,:1])(img_input),conv4])
    
    model= Model(input = img_input, output = added)
    
    #plot_model(model, to_file='./model_png/convLSTM_res_attention.png', show_shapes=True, show_layer_names=True)
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    print(model.summary())
    
    if optimizer =='adam':
        opt = keras.optimizers.Adam(lr=lr, decay=decay)
        
    model.compile(optimizer = opt, loss = loss)
    return model

def convLSTM_attention_pred(timesteps=None, input_width=224, input_height=224, input_channel=4,
                  multi_gpu = False, ngpu=1,
                  optimizer='adam', loss='mean_squared_error',
                  lr = 0.001, decay = 0.00001) :
    
    #input layer
    img_input = Input(shape=(timesteps, None, None, input_channel))
    flow = Lambda(lambda x: x[:,1:,:,:,1:])(img_input)
    
    # unet block1
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(flow)
    conv1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv1)
    conv1 = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True)(conv1)
    add1 = multiply([Lambda(lambda x: x[:,1:,:,:,:1])(img_input),conv1])

    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(add1)
    conv2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    concat2 = concatenate([add1,conv2])
    conv2 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat2)
    
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    conv3 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv3)
    concat3 = concatenate([conv2,conv3])
    conv3 = ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same', return_sequences=True)(concat3)
    
    conv4 = ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=False)(conv3)
    
    #added = add([Lambda(lambda x: x[:,-1,:,:,:1])(img_input),conv4])
    
    model= Model(input = img_input, output = conv4)
    
    #plot_model(model, to_file='./model_png/convLSTM_res_attention.png', show_shapes=True, show_layer_names=True)
    
    if multi_gpu:
        model = keras.utils.multi_gpu_model(model, gpus=ngpu)
    print(model.summary())
    
    if optimizer =='adam':
        opt = keras.optimizers.Adam(lr=lr, decay=decay)
        
    if loss == 'DSSIM':
        loss = DSSIMObjective(kernel_size=4)
        
    model.compile(optimizer = opt, loss = loss)
    return model


