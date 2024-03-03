import cv2 
import os 

#train

cam = cv2.VideoCapture('data/subway/enter/subway_entrance_turnstiles.AVI')
try: 
    # creating a folder named data 
    if not os.path.exists('data/subway/enter/all_frames/'): 
        os.makedirs('data/subway/enter/all_frames/') 
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
        name = 'data/subway/enter/all_frames/'+ str(currentframe) + '.jpg'
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
        