from utils.models import *
from utils.data_generator import *
import argparse
import numpy as np
import os
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

def parse_args():
    parser = argparse.ArgumentParser(description = 'training')
   
    # dataset
    parser.add_argument('--dataset', type = str, default = "avenue", help = 'dataset name')
    parser.add_argument('--input_channel', type = int, default = 1, help = 'input_channel size')
    parser.add_argument('--image_size', type=int, default=224, help='image resize size')
    parser.add_argument('--timesteps', type = int, default = 3, help = 'timesteps(window size)')
    
    # model
    parser.add_argument('--model', type = str, default = 'convLSTM_unet1', help = 'model name')
    parser.add_argument('--multi_gpu', action='store_true', default=False , help = 'use multi gpu')
    parser.add_argument('--ngpus', type = int, default = 1, help = 'number of GPUs')
    
    # model compile
    parser.add_argument('--optimizer', type = str, default = 'adam', help = 'optimizer')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
    parser.add_argument('--decay', type = float, default = 0.00001, help = 'learning rate decay')
    parser.add_argument('--loss', type = str, default = 'mean_squared_error', help = 'loss function')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'training batch size')
    
    # traiing
    parser.add_argument('--epochs', type = int, default = 100, help = 'training epochs')
    
    # path
    parser.add_argument('--checkname', type = str, default = False, help = 'save dir name')
    
    args = parser.parse_args()
    print(args)
    return args
       
        
if __name__ == "__main__":
    
    # parser
    args = parse_args()
    
    # gpu setting
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]= args.CUDA_VISIBLE_DEVICES

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    
    
    # dataset generator
#    generator = avenue_train_generator_new(train_videos_path = './data/Avenue Dataset/training_frames/', 
#                           timesteps = args.timesteps, image_size = args.image_size, batch_size = args.batch_size)
#    generator = avenue_train_generator_flow(train_videos_path = './data/Avenue Dataset/training_frames/', 
#                           timesteps = args.timesteps, image_size = args.image_size, batch_size = args.batch_size)    
    

#    generator = UCSD_train_generator_new(dataset = args.dataset, 
#                           timesteps = args.timesteps, image_size = args.image_size, batch_size = args.batch_size)
#    generator = UCSD_train_generator_flow(dataset = args.dataset, 
#                           timesteps = args.timesteps, image_size = args.image_size, batch_size = args.batch_size)

#    generator = UCSD_train_generator_res(dataset = args.dataset,
#                                         timesteps = args.timesteps, image_size = args.image_size, batch_size = args.batch_size)
#    generator = UCSD_train_generator_res_flow(dataset = args.dataset, 
#                           timesteps = args.timesteps, image_size = args.image_size, batch_size = args.batch_size)
#    generator = UCSD_res_flow(dataset = args.dataset, 
#                           timesteps = args.timesteps, image_size = args.image_size, batch_size = args.batch_size)    
#    generator = res_flow_generator(dataset = args.dataset, 
#                       timesteps = args.timesteps, image_size = args.image_size, batch_size = args.batch_size)    
    generator = res_flow_generator_pred(dataset = args.dataset, 
                       timesteps = args.timesteps, image_size = args.image_size, batch_size = args.batch_size) 
#    generator = res_flow_generator_auto(dataset = args.dataset, 
#                       timesteps = args.timesteps, image_size = args.image_size, batch_size = args.batch_size)
#    generator = res_flow_generator_stack(dataset = args.dataset, 
#                       timesteps = args.timesteps, image_size = args.image_size, batch_size = args.batch_size)
    
    
    # model load
    model = eval(args.model + '(timesteps = args.timesteps, input_width=args.image_size, \
                 input_height=args.image_size, input_channel=args.input_channel, \
                 multi_gpu = args.multi_gpu, ngpu=args.ngpus,\
                 optimizer = args.optimizer, loss = args.loss, lr = args.lr, decay = args.decay)')
    
    dir_path = './runs/dataset_{}_image_size_{}_timesteps_{}_epochs_{}_model_{}_checkname_{}'.format(args.dataset, args.image_size, args.timesteps, args.epochs, args.model, args.checkname)
    checkname = args.checkname
    
    if not os.path.exists(dir_path):
        os.mkdir(dir_path) 
        
    if not os.path.exists(dir_path +'/' + checkname):    
        os.mkdir(dir_path +'/' + checkname)
        
    filepath = dir_path +'/'+ checkname + "/{epoch:03d}-{loss:.8f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False)
    early_stopping  = EarlyStopping(monitor='loss', patience=100, mode='min')
    tb_hist = keras.callbacks.TensorBoard(log_dir='./graph/dataset_{}_image_size_{}_timesteps_{}_epochs_{}_model_{}_checkname_{}'.format(args.dataset, args.image_size,args.timesteps, args.epochs, args.model, args.checkname), histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, early_stopping, tb_hist]

    model.fit_generator(generator,epochs=args.epochs,callbacks = callbacks_list, steps_per_epoch=500)
    