import cv2 
import os 

#train
for file in os.listdir('data/subway/enter/training_videos/'):
    if file.split(".")[-1] == 'avi':
        
        cam = cv2.VideoCapture('data/subway/enter/training_videos/' + file)
        try: 
            # creating a folder named data 
            if not os.path.exists('data/subway/enter/training_frames/'+str(file.split(".")[0])): 
                os.makedirs('data/subway/enter/training_frames/'+str(file.split(".")[0])) 
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
                name = 'data/subway/enter/training_frames/'+ str(file.split(".")[0]) +'/'+ str(currentframe) + '.jpg'
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
        
for file in os.listdir('data/subway/exit/training_videos/'):
    if file.split(".")[-1] == 'avi':
        
        cam = cv2.VideoCapture('data/subway/exit/training_videos/' + file)
        try: 
            # creating a folder named data 
            if not os.path.exists('data/subway/exit/training_frames/'+str(file.split(".")[0])): 
                os.makedirs('data/subway/exit/training_frames/'+str(file.split(".")[0])) 
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
                name = 'data/subway/exit/training_frames/'+ str(file.split(".")[0]) +'/'+ str(currentframe) + '.jpg'
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
for file in os.listdir('data/subway/enter/testing_videos/'):
    if file.split(".")[-1] == 'avi':
        
        cam = cv2.VideoCapture('data/subway/enter/testing_videos/' + file)
        try: 
            # creating a folder named data 
            if not os.path.exists('data/subway/enter/testing_frames/'+str(file.split(".")[0])): 
                os.makedirs('data/subway/enter/testing_frames/'+str(file.split(".")[0])) 
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
                name = 'data/subway/enter/testing_frames/'+ str(file.split(".")[0]) +'/'+ str(currentframe) + '.jpg'
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
        
for file in os.listdir('data/subway/exit/testing_videos/'):
    if file.split(".")[-1] == 'avi':
        
        cam = cv2.VideoCapture('data/subway/exit/testing_videos/' + file)
        try: 
            # creating a folder named data 
            if not os.path.exists('data/subway/exit/testing_frames/'+str(file.split(".")[0])): 
                os.makedirs('data/subway/exit/testing_frames/'+str(file.split(".")[0])) 
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
                name = 'data/subway/exit/testing_frames/'+ str(file.split(".")[0]) +'/'+ str(currentframe) + '.jpg'
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