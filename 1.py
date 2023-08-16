'''
# CODE FOR ENABLING CAMERA
import cv2

# Open the default camera (camera index 0)
cam = cv2.VideoCapture(0)
while True:
    ret, video_data = cam.read()  
    cv2.imshow("video_live", video_data)  
    # Wait for a key press, and if 'a' is pressed, exit the loop
    if cv2.waitKey(10) == ord("a"):
        break

# Release the camera and close all OpenCV windows
cam.release()
'''
import os
import numpy as np
import cv2
from PIL import Image

cam = cv2.VideoCapture(0)
cam.set(3,640) # set Width
cam.set(4,480) # set Height
# Load the cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')    
    # For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")   
    # Initialize individual sampling face count
count = 0
while True:
    ret, video_data = cam.read()  
    cv2.flip(video_data,-1)  # flip video image vertically
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)   
    for (x,y,w,h) in faces:
         cv2.rectangle(video_data, (x,y), (x+w,y+h), (255,0,0), 2) 
         count += 1
        # Save the captured image into the datasets folder
         cv2.imwrite("Dataset/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
         cv2.imshow('image', video_data)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 100: # Take 30 face sample and stop video
         break    
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

# Path for face image database
path = 'FaceDetection/Dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)




# Load training data here (populate 'faces' and 'ids' lists)
if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
else:
    print("No training data available.")

# Continue with your recognition process



# Save the model into trainer/trainer.yml
recognizer.write('Trainer/trainer.yml') 
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    
    
