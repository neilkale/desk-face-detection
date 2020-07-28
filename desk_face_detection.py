import cv2
import time
from datetime import date
import atexit
import ast

COUNTING_TIME = 10
WAITING_TIME = 120
MIN_CONF = 0.5

frames_with_face = 0
frames = 0
timeAtDesk = 0
initEnd = False
today = date.today().strftime('%m/%d/%Y')

frontal_face_default = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

file = open('output.txt','r')
contents = file.read()
dictionary = ast.literal_eval(contents) #this is more safe than eval(), in case the string is dangerous
file.close()

if(today in dictionary):
    timeAtDesk = dictionary[today]*60
    pass

def save_data():
    dictionary[today] = round(timeAtDesk/60, 2)
    file = open('output.txt', 'w')
    file.write(str(dictionary))
    file.close()
    pass
atexit.register(save_data)

cap = cv2.VideoCapture(0)

while(True):
    while(time.time() < time_end):
        ret, frame = cap.read()
        #pre-processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (13,13), 0)
        #finding faces in the frame and displaying them
        faces = frontal_face_default.detectMultiScale(gray, 1.2, 2)
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame, (x,y), (x + w,y + h), (0,0,255),2)
            pass
        cv2.putText(frame, "You have been sitting here for: {0:.2f} minutes".format(timeAtDesk/60), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (90,50,190), 2)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '): #Why do I need this line, and time.sleep(0.001) doesn't work???
            initEnd = True
            break
        #collecting data to give a likelihood of face existence
        frames += 1
        if(len(faces) != 0): frames_with_face += 1
        pass
    if(frames != 0 and frames_with_face / frames > MIN_CONF): 
        timeAtDesk += (COUNTING_TIME)
    frames = 0
    frames_with_face = 0
    if(initEnd):
        break

cap.release()

#Code that uses the wait time. The frame stops responding during wait time, so this code isn't usable right now.
while(False):
    #Time (s) between counts, so the webcam doesn't overheat due to running
    #constantly
    time_end = time.time() + WAITING_TIME
    while(time.time() < time_end):
        pass
    
    #Starting the webcam
    cap = cv2.VideoCapture(0)
    
    #The length (s) of each sampling.
    time_end = time.time() + COUNTING_TIME
    while(time.time() < time_end):
        ret, frame = cap.read()
        #pre-processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (13,13), 0)
        #finding faces in the frame and displaying them
        faces = frontal_face_default.detectMultiScale(gray, 1.2, 2)
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame, (x,y), (x + w,y + h), (0,0,255),2)
            pass
        cv2.putText(frame, "You have been sitting here for: {0:.2f} minutes".format(timeAtDesk/60), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (90,50,190), 2)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '): #Why do I need this line, and time.sleep(0.001) doesn't work???
            initEnd = True
            break
        #collecting data to give a likelihood of face existence
        frames += 1
        if(len(faces) != 0): frames_with_face += 1
        pass

    #If faces were present in more than MIN_CONF of the frames, we assume you
        #were at desk for the between time as well.
    if(frames != 0 and frames_with_face / frames > MIN_CONF): 
        timeAtDesk += (COUNTING_TIME + WAITING_TIME)
    frames = 0
    frames_with_face = 0
    
    #Stopping the webcam
    cap.release()

    if(initEnd):
            break
