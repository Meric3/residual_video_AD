import cv2 
import os 

#train
for file in os.listdir('data/Avenue Dataset/training_videos/'):
    if file.split(".")[-1] == 'avi':
        
        cam = cv2.VideoCapture('data/Avenue Dataset/training_videos/' + file)
        try: 
            # creating a folder named data 
            if not os.path.exists('data/Avenue Dataset/training_frames/'+str(file.split(".")[0])): 
                os.makedirs('data/Avenue Dataset/training_frames/'+str(file.split(".")[0])) 
        # if not created then raise error 
        except OSError: 
            print ('Error: Creating directory of data') 

        # frame 
        currentframe = 0

        while(True): 
            # reading from frame 
            ret,frame = cam.read() 
            if ret: 
                # if video is still left continue creating images 
                name = 'data/Avenue Dataset/training_frames/'+ str(file.split(".")[0]) +'/'+ str(currentframe) + '.jpg'
                print ('Creating...' + name) 

                # writing the extracted images 
                cv2.imwrite(name, frame) 

                # increasing counter so that it will 
                # show how many frames are created 
                currentframe += 1
            else: 
                break

        # Release all space and windows once done 
        cam.release() 
        cv2.destroyAllWindows() 
        

#test
for file in os.listdir('data/Avenue Dataset/testing_videos/'):
    if file.split(".")[-1] == 'avi':
        
        cam = cv2.VideoCapture('data/Avenue Dataset/testing_videos/' + file)
        try: 
            # creating a folder named data 
            if not os.path.exists('data/Avenue Dataset/testing_frames/'+str(file.split(".")[0])): 
                os.makedirs('data/Avenue Dataset/testing_frames/'+str(file.split(".")[0])) 
        # if not created then raise error 
        except OSError: 
            print ('Error: Creating directory of data') 

        # frame 
        currentframe = 0

        while(True): 
            # reading from frame 
            ret,frame = cam.read() 
            if ret: 
                # if video is still left continue creating images 
                name = 'data/Avenue Dataset/testing_frames/'+ str(file.split(".")[0]) +'/'+ str(currentframe) + '.jpg'
                print ('Creating...' + name) 

                # writing the extracted images 
                cv2.imwrite(name, frame) 

                # increasing counter so that it will 
                # show how many frames are created 
                currentframe += 1
            else: 
                break

        # Release all space and windows once done 
        cam.release() 
        cv2.destroyAllWindows() 