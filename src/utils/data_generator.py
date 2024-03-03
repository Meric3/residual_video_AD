import os
import numpy as np
import scipy.io
import cv2
import pdb
      
def avenue_train_generator_new(train_videos_path = './data/Avenue Dataset/training_frames/', 
                           timesteps = None, image_size = 224, batch_size = None): 
    
    videos = dict()
    for video in np.sort(os.listdir(train_videos_path)) :
        print('check video name : ', video+'.avi')

        # resize
        resized_frames =[]
        for name in range(len(os.listdir(os.path.join(train_videos_path, video)))):
            frame = scipy.misc.imread(os.path.join(train_videos_path,video,str(name))+'.jpg', mode='L') # 'L = gray'
            frame = scipy.misc.imresize(frame, (image_size,image_size))
            resized_frames.append(frame)

        # stacking 
        resized_frames = np.stack(resized_frames, axis=0)
        videos[str(video)] = resized_frames/255.        
    print(list(videos.keys()))
    while True:
        batch = []

        for _ in range(batch_size):
            #random video load
            video_name = np.random.choice(list(videos.keys()))
            video_frames = videos[video_name]
            video_length = video_frames.shape[0]
            sequence_start_idx = np.random.choice(range(video_length-(timesteps-1)))
            sequence = video_frames[sequence_start_idx:sequence_start_idx+timesteps,:,:]

            # add batch
            batch.append(sequence)
        batch = np.stack(batch, axis = 0) 
        batch = np.expand_dims(batch, -1)
        yield batch, batch        
        
def avenue_train_generator_flow(train_videos_path = './data/Avenue Dataset/training_frames/', 
                           timesteps = None, image_size = 224, batch_size = None): 
    
    videos = dict()
    for video in np.sort(os.listdir(train_videos_path)) :
        print('check video name : ', video+'.avi')

        # resize
        resized_frames =[]

        prv = scipy.misc.imread(os.path.join(train_videos_path,video,str(0))+'.jpg', mode='L')
        prv = scipy.misc.imresize(prv, (image_size,image_size))

        for name in range(1,len(os.listdir(os.path.join(train_videos_path, video)))):
            nxt = scipy.misc.imread(os.path.join(train_videos_path,video,str(name))+'.jpg', mode='L') # 'L = gray'
            nxt = scipy.misc.imresize(nxt, (image_size,image_size))

            flow = cv2.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

            frame = np.stack([nxt/255.,cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX),ang/np.pi/2],axis = -1)

            resized_frames.append(frame)
            prv = nxt.copy()

        # stacking 
        resized_frames = np.stack(resized_frames, axis=0)
        videos[str(video)] = resized_frames        
    print(list(videos.keys()))
    
    while True:
        batch = []

        for _ in range(batch_size):
            #random video load
            video_name = np.random.choice(list(videos.keys()))
            video_frames = videos[video_name]
            video_length = video_frames.shape[0]
            sequence_start_idx = np.random.choice(range(video_length-(timesteps-1)))
            sequence = video_frames[sequence_start_idx:sequence_start_idx+timesteps,:,:,:]

            # add batch
            batch.append(sequence)
        batch = np.stack(batch, axis = 0) 
        yield batch, batch[...,:1]  
        
        
def avenue_train_generator_auto_new(train_videos_path = './data/Avenue Dataset/training_frames/', 
                           timesteps = None, image_size = 224, batch_size = None): 
    
    videos = dict()
    for video in np.sort(os.listdir(train_videos_path)) :
        print('video name : ', video+'.avi')

        # resize
        resized_frames =[]
        for name in range(len(os.listdir(os.path.join(train_videos_path, video)))):
            frame = scipy.misc.imread(os.path.join(train_videos_path,video,str(name))+'.jpg', mode='L') # 'L = gray'
            frame = scipy.misc.imresize(frame, (image_size,image_size))
            resized_frames.append(frame)

        # stacking 
        resized_frames = np.stack(resized_frames, axis=0)
        videos[str(video)] = resized_frames/255.        
    print(list(videos.keys()))
    
    while True:
        batch = []
        
        for _ in range(batch_size):
            #random video load
            video_name = np.random.choice(list(videos.keys()))
            video_frames = videos[video_name]
            video_length = video_frames.shape[0]
            sequence_start_idx = np.random.choice(range(video_length-(timesteps-1)))
            sequence = video_frames[sequence_start_idx:sequence_start_idx+timesteps,:,:]
 
            # add batch
            batch.append(sequence)
        batch = np.stack(batch, axis = 0) 
        batch = np.expand_dims(batch, -1) #(b,t,w,h,c)
        
        reverse_batch = np.flip(batch, 1)
        yield [batch, reverse_batch], reverse_batch            

        
def UCSD_train_generator_new(dataset = 'ped1', timesteps = 3, image_size = 224, batch_size = None): 
    if dataset == 'ped1' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'
    elif dataset == 'ped2' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
        
 
    train_video_list = np.sort([folder for folder in os.listdir(os.path.join(train_videos_path,'Train')) if 'Train' in folder])

    while True:
        batch = []

        for _ in range(batch_size):
            video_name = np.random.choice(train_video_list)
            video_frames = np.sort([file for file in os.listdir(os.path.join(train_videos_path,'Train',video_name)) if 'tif' in file])
            video_length = len(video_frames)
            sequence_start_idx = np.random.choice(range(video_length-(timesteps-1)))
            sequence = video_frames[sequence_start_idx:sequence_start_idx+timesteps]

            resized_frames = []
            for name in sequence :
                frame = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, name), mode='L') # 'L = gray'
                frame = scipy.misc.imresize(frame, (image_size,image_size))
                resized_frames.append(frame)

            resized_frames = np.stack(resized_frames, axis=0)/255.
            batch.append(resized_frames)

        batch = np.stack(batch, axis = 0) 
        batch = np.expand_dims(batch, -1)
        yield batch, batch        

        
def UCSD_train_generator_flow(dataset = 'ped1', timesteps = 3, image_size = 224, batch_size = None): 
    if dataset == 'ped1' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'
    elif dataset == 'ped2' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
        
 
    train_video_list = np.sort([folder for folder in os.listdir(os.path.join(train_videos_path,'Train')) if 'Train' in folder])

    while True:
        batch = []

        for _ in range(batch_size):
            video_name = np.random.choice(train_video_list)
            video_frames = np.sort([file for file in os.listdir(os.path.join(train_videos_path,'Train',video_name)) if 'tif' in file])
            video_length = len(video_frames)
            sequence_start_idx = np.random.choice(range(1,video_length-(timesteps-1)))
            sequence = video_frames[sequence_start_idx-1:sequence_start_idx+timesteps]

            resized_frames = []
            
            prv = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, sequence[0]), mode='L')
            prv = scipy.misc.imresize(prv, (image_size,image_size))
            
            for name in sequence[1:] :
                nxt = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, name), mode='L') # 'L = gray'
                nxt = scipy.misc.imresize(nxt, (image_size,image_size))

                flow = cv2.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                
                frame = np.stack([nxt/255.,cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX),ang/np.pi/2],axis = -1)
                resized_frames.append(frame)
                prv = nxt.copy()

            resized_frames = np.stack(resized_frames, axis=0)
            batch.append(resized_frames)

        batch = np.stack(batch, axis = 0) 

        yield batch, batch[...,:1]  
        
def UCSD_train_generator_auto(dataset = 'ped1', timesteps = 3, image_size = 224, batch_size = None): 
    if dataset == 'ped1' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'
    elif dataset == 'ped2' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
        
 
    train_video_list = np.sort([folder for folder in os.listdir(os.path.join(train_videos_path,'Train')) if 'Train' in folder])

    while True:
        batch = []

        for _ in range(batch_size):
            video_name = np.random.choice(train_video_list)
            video_frames = np.sort([file for file in os.listdir(os.path.join(train_videos_path,'Train',video_name)) if 'tif' in file])
            video_length = len(video_frames)
            sequence_start_idx = np.random.choice(range(video_length-(timesteps-1)))
            sequence = video_frames[sequence_start_idx:sequence_start_idx+timesteps]

            resized_frames = []
            for name in sequence :
                frame = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, name), mode='L') # 'L = gray'
                frame = scipy.misc.imresize(frame, (image_size,image_size))
                resized_frames.append(frame)

            resized_frames = np.stack(resized_frames, axis=0)/255.
            batch.append(resized_frames)

        batch = np.stack(batch, axis = 0) 
        batch = np.expand_dims(batch, -1)
        
        reverse_batch = np.flip(batch, 1)
        yield [batch, reverse_batch], reverse_batch    

def UCSD_train_generator_res(dataset = 'ped1', timesteps = 3, image_size = 224, batch_size = None): 
    if dataset == 'ped1' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'
    elif dataset == 'ped2' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
        
 
    train_video_list = np.sort([folder for folder in os.listdir(os.path.join(train_videos_path,'Train')) if 'Train' in folder])

    while True:
        batch = []

        for _ in range(batch_size):
            video_name = np.random.choice(train_video_list)
            video_frames = np.sort([file for file in os.listdir(os.path.join(train_videos_path,'Train',video_name)) if 'tif' in file])
            video_length = len(video_frames)
            sequence_start_idx = np.random.choice(range(video_length-(timesteps-1)))
            sequence = video_frames[sequence_start_idx:sequence_start_idx+timesteps]

            resized_frames = []
            for name in sequence :
                frame = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, name), mode='L') # 'L = gray'
                frame = scipy.misc.imresize(frame, (image_size,image_size))
                resized_frames.append(frame)

            resized_frames = np.stack(resized_frames, axis=0)/255.
            batch.append(resized_frames)

        batch = np.stack(batch, axis = 0) 
        batch = np.expand_dims(batch, -1)
        yield batch, batch[:,1:,:,:,:]     
        
def UCSD_train_generator_res_flow(dataset = 'ped1', timesteps = 3, image_size = 224, batch_size = None): 
    if dataset == 'ped1' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'
    elif dataset == 'ped2' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
        
 
    train_video_list = np.sort([folder for folder in os.listdir(os.path.join(train_videos_path,'Train')) if 'Train' in folder])

    while True:
        batch = []

        for _ in range(batch_size):
            video_name = np.random.choice(train_video_list)
            video_frames = np.sort([file for file in os.listdir(os.path.join(train_videos_path,'Train',video_name)) if 'tif' in file])
            video_length = len(video_frames)
            sequence_start_idx = np.random.choice(range(1,video_length-(timesteps-1)))
            sequence = video_frames[sequence_start_idx-1:sequence_start_idx+timesteps]

            resized_frames = []
            
            prv = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, sequence[0]), mode='L')
            prv = scipy.misc.imresize(prv, (image_size,image_size))
            
            for name in sequence[1:] :
                nxt = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, name), mode='L') # 'L = gray'
                nxt = scipy.misc.imresize(nxt, (image_size,image_size))

                flow = cv2.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                
                frame = np.stack([nxt/255.,cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX),ang/np.pi/2],axis = -1)
                resized_frames.append(frame)
                prv = nxt.copy()

            resized_frames = np.stack(resized_frames, axis=0)
            batch.append(resized_frames)

        batch = np.stack(batch, axis = 0) 

        yield batch, batch[:,1:,:,:,:1]        
        
def UCSD_res_flow(dataset = 'ped1', timesteps = 3, image_size = 224, batch_size = None): 
    if dataset == 'ped1' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'
    elif dataset == 'ped2' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
        
 
    train_video_list = np.sort([folder for folder in os.listdir(os.path.join(train_videos_path,'Train')) if 'Train' in folder])

    while True:
        batch = []

        for _ in range(batch_size):
            video_name = np.random.choice(train_video_list)
            video_frames = np.sort([file for file in os.listdir(os.path.join(train_videos_path,'Train',video_name)) if 'tif' in file])
            video_length = len(video_frames)
            sequence_start_idx = np.random.choice(range(1,video_length-(timesteps-1)))
            sequence = video_frames[sequence_start_idx-1:sequence_start_idx+timesteps]

            resized_frames = []
            
            prv = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, sequence[0]), mode='L')
            prv = scipy.misc.imresize(prv, (image_size,image_size))
            
            for name in sequence[1:] :
                nxt = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, name), mode='L') # 'L = gray'
                nxt = scipy.misc.imresize(nxt, (image_size,image_size))

                flow = cv2.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                hsv = np.zeros((image_size,image_size,3))
                hsv[...,1] = 255.  
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                hsv = np.uint8(hsv) 
                
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
                rgb = rgb/255.
                
                nxt = np.expand_dims(nxt, axis=-1)
                frame = np.concatenate([nxt/255.,rgb],axis = -1)
                resized_frames.append(frame)
                prv = nxt.copy()

            resized_frames = np.stack(resized_frames, axis=0)
            batch.append(resized_frames)

        batch = np.stack(batch, axis = 0) 

        yield batch, batch[:,1:,:,:,:1]   
        
        
def UCSD_res_flow_crop(dataset = 'ped1', timesteps = 3, image_size = 224, batch_size = None): 
    if dataset == 'ped1' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'
    elif dataset == 'ped2' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
        
 
    train_video_list = np.sort([folder for folder in os.listdir(os.path.join(train_videos_path,'Train')) if 'Train' in folder])

    while True:
        batch = []

        for _ in range(batch_size):
            video_name = np.random.choice(train_video_list)
            video_frames = np.sort([file for file in os.listdir(os.path.join(train_videos_path,'Train',video_name)) if 'tif' in file])
            video_length = len(video_frames)
            sequence_start_idx = np.random.choice(range(1,video_length-(timesteps-1)))
            sequence = video_frames[sequence_start_idx-1:sequence_start_idx+timesteps]

            resized_frames = []
            
            idx =  tuple(np.random.choice(range(2)) for _ in range(2)) #crop idx
            prv = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, sequence[0]), mode='L')
            height, width = prv.shape[0], prv.shape[1]

            for name in sequence[1:] :
                nxt = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, name), mode='L') # 'L = gray'

                flow = cv2.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                hsv = np.zeros((height,width,3))
                hsv[...,1] = 255.  
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                hsv = np.uint8(hsv) 
                
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
                rgb = rgb
                
                rgb = rgb[height//2*idx[0]:(height//2*idx[0] + height//2), width//2*idx[1]:(width//2*idx[1] + width//2)] #crop
                rgb = scipy.misc.imresize(rgb, (image_size,image_size))
                
                prv=nxt.copy()
                nxt = nxt[height//2*idx[0]:(height//2*idx[0] + height//2), width//2*idx[1]:(width//2*idx[1] + width//2)] #crop
                nxt = scipy.misc.imresize(nxt, (image_size,image_size))
                
                nxt = np.expand_dims(nxt, axis=-1)
                frame = np.concatenate([nxt/255.,rgb/255.],axis = -1)
                resized_frames.append(frame)
            resized_frames = np.stack(resized_frames, axis=0)
            batch.append(resized_frames)

        batch = np.stack(batch, axis = 0) 

        yield batch, batch[:,1:,:,:,:1]    
        
        
def res_flow_generator(dataset = 'avenue', timesteps = 3, image_size = 224, batch_size = None): 
    
    if dataset =='avenue':
        train_videos_path = './data/Avenue Dataset/training_frames/'
    elif dataset == 'ped1' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'
    elif dataset == 'ped2' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
    elif dataset == 'enter' :
        train_videos_path = './data/subway/enter/training_frames/'
    elif dataset == 'exit' :
        train_videos_path = './data/subway/exit/training_frames/'
        
    if dataset =='avenue' or dataset == 'enter' or dataset == 'exit':
        train_video_list = np.sort(os.listdir(train_videos_path))
    elif dataset == 'ped1' or dataset == 'ped2':
        train_video_list = np.sort([folder for folder in os.listdir(os.path.join(train_videos_path,'Train')) if 'Train' in folder])

    while True:
        batch = []

        for _ in range(batch_size):
            video_name = np.random.choice(train_video_list)
            
            if dataset =='avenue' or dataset == 'enter' or dataset == 'exit':
                video_length = len(os.listdir(os.path.join(train_videos_path,video_name)))
                video_frames = [str(file) + '.jpg' for file in list(range(video_length))]
                
            elif dataset == 'ped1' or dataset == 'ped2':    
                video_frames = np.sort([file for file in os.listdir(os.path.join(train_videos_path,'Train',video_name)) if 'tif' in file])
                video_length = len(video_frames)
                
            sequence_start_idx = np.random.choice(range(1,video_length-(timesteps-1)))
            sequence = video_frames[sequence_start_idx-1:sequence_start_idx+timesteps]

            resized_frames = []
            
            if dataset =='avenue' or dataset == 'enter' or dataset == 'exit':
                prv = scipy.misc.imread(os.path.join(train_videos_path,video_name,sequence[0]), mode='L')
                
            elif dataset == 'ped1' or dataset == 'ped2':
                prv = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, sequence[0]), mode='L')
           
            prv = scipy.misc.imresize(prv, (image_size,image_size))
            
            for name in sequence[1:] :
                if dataset =='avenue' or dataset == 'enter' or dataset == 'exit':
                    nxt = scipy.misc.imread(os.path.join(train_videos_path,video_name,name), mode='L')
                elif dataset == 'ped1' or dataset == 'ped2':
                    nxt = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, name), mode='L') # 'L = gray'
                    
                nxt = scipy.misc.imresize(nxt, (image_size,image_size))

                flow = cv2.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                hsv = np.zeros((image_size,image_size,3))
                hsv[...,1] = 255.  
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                hsv = np.uint8(hsv) 
                
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
                rgb = rgb/255.
                
                nxt = np.expand_dims(nxt, axis=-1)
                frame = np.concatenate([nxt/255.,rgb],axis = -1)
                resized_frames.append(frame)
                prv = nxt.copy()

            resized_frames = np.stack(resized_frames, axis=0)
            batch.append(resized_frames)

        batch = np.stack(batch, axis = 0) 

        yield batch, batch[:,1:,:,:,:1]   
        
def res_flow_generator_pred(dataset = 'avenue', timesteps = 3, image_size = 224, batch_size = None): 
    
    
    if dataset =='avenue':
        train_videos_path = './data/Avenue Dataset/training_frames/'
    elif dataset == 'ped1' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/'
    elif dataset == 'ped2' :
        train_videos_path = './data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
    elif dataset == 'enter' :
        train_videos_path = './data/subway/enter/training_frames/'
    elif dataset == 'exit' :
        train_videos_path = './data/subway/exit/training_frames/'
    elif dataset == 'shang' :
        train_videos_path = './data/shanghaitech/training/frames/'
        
        
    if dataset =='avenue' or dataset == 'enter' or dataset == 'exit':
        train_video_list = np.sort(os.listdir(train_videos_path))
    elif dataset == 'ped1' or dataset == 'ped2':
        train_video_list = np.sort([folder for folder in os.listdir(os.path.join(train_videos_path,'Train')) if 'Train' in folder])
    elif dataset == 'shang':
#         train_video_list = np.sort([folder for folder in os.listdir(os.path.join(train_videos_path,'frames')) if 'frames' in folder])
        train_video_list = np.sort(os.listdir(train_videos_path))

        

    while True:
        batch = []
        batch_y = []
        for _ in range(batch_size):
            video_name = np.random.choice(train_video_list)
            
            if dataset =='avenue' or dataset == 'enter' or dataset == 'exit' or dataset == 'shang':
                video_length = len(os.listdir(os.path.join(train_videos_path, video_name)))
                video_frames = [str(file) + '.jpg' for file in list(range(video_length))]
                
            elif dataset == 'ped1' or dataset == 'ped2':    
                video_frames = np.sort([file for file in os.listdir(os.path.join(train_videos_path,'Train',video_name)) if 'tif' in file])
                video_length = len(video_frames)
                
            
                
            sequence_start_idx = np.random.choice(range(1,video_length-(timesteps-1)-1))
            sequence = video_frames[sequence_start_idx-1:sequence_start_idx+timesteps]
            
            
            resized_frames = []
            if dataset =='avenue' or dataset == 'enter' or dataset == 'exit' or dataset == 'shang':
                frame_y = scipy.misc.imread(os.path.join(train_videos_path,video_name,video_frames[sequence_start_idx+timesteps]), mode='L')
                
            elif dataset == 'ped1' or dataset == 'ped2':
                frame_y = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name,video_frames[sequence_start_idx+timesteps]), mode='L')
                
            frame_y = scipy.misc.imresize(frame_y, (image_size,image_size))
            frame_y = np.expand_dims(frame_y, axis=-1)
            
            if dataset =='avenue' or dataset == 'enter' or dataset == 'exit' or dataset == 'shang':
                prv = scipy.misc.imread(os.path.join(train_videos_path,video_name,sequence[0]), mode='L')
                
            elif dataset == 'ped1' or dataset == 'ped2':
                prv = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, sequence[0]), mode='L')
           
            prv = scipy.misc.imresize(prv, (image_size,image_size))
            
#             pdb.set_trace()
            
            for name in sequence[1:] :
                if dataset =='avenue' or dataset == 'enter' or dataset == 'exit' or dataset == 'shang':
                    nxt = scipy.misc.imread(os.path.join(train_videos_path,video_name,name), mode='L')
                elif dataset == 'ped1' or dataset == 'ped2':
                    nxt = scipy.misc.imread(os.path.join(train_videos_path,'Train',video_name, name), mode='L') # 'L = gray'
                    
                nxt = scipy.misc.imresize(nxt, (image_size,image_size))

                flow = cv2.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

                hsv = np.zeros((image_size,image_size,3))
                hsv[...,1] = 255.  
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                hsv = np.uint8(hsv) 
                
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
                rgb = rgb/255.
                
                nxt = np.expand_dims(nxt, axis=-1)
                frame = np.concatenate([nxt/255.,rgb],axis = -1)
                resized_frames.append(frame)
                prv = nxt.copy()

            resized_frames = np.stack(resized_frames, axis=0)
            batch.append(resized_frames)
            batch_y.append(frame_y/255.)
            
            
        batch = np.stack(batch, axis = 0) 
        batch_y = np.stack(batch_y, axis = 0)
        
        yield batch, batch_y  
        


