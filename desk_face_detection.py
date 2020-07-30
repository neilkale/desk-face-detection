import cv2
import time
import datetime as dt
import atexit
import ast
import matplotlib.pyplot as plt
import os
import numpy as np

TIME_INTERVAL = 1
NUM_FRAMES = 1000
MIN_CONF = 0.5

frames_with_face = 0
frames = 0
timeAtDesk = 0
initEnd = False
today = dt.date.today().strftime('%m_%d_%Y')
frontal_face_default = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Making the output/input folders if they don't exist
try:
    os.mkdir('daily_data')
except Exception:
    pass

try:
    os.mkdir('daily_graphs')
except Exception:
    pass

dataPath = "daily_data/"+today+".txt"
try:
    data_file = open(dataPath,'x')
    data_file.close()
except:
    pass

#Making/opening the input file
data_file = open(dataPath, 'r')
contents = data_file.read()
try:
    to_graph = ast.literal_eval(contents) #this is more safe than eval(), in case the string is dangerous
except:
    to_graph = {}
data_file.close()

#To be called atexit
def save_data():
    data_file = open("daily_data/"+today+".txt",'w')
    data_file.write(str(to_graph))
    data_file.close()

    graph_file = open("daily_graphs/"+today+".jpg",'wb')

    times = ["{}:{}:{}".format(t[3],t[4],t[5]) for t in to_graph.keys()]
    activity = [val for val in to_graph.values()]
    total_time_mins = sum(activity)*TIME_INTERVAL/60

    fig, ax = plt.subplots()

    ax.plot_date(times,activity, linestyle = "-", marker = "", drawstyle='steps-post')
    ax.fill_between(times,activity,step='post',alpha=0.4)

    n = int(len(times)/10)+1
    ax.set_xticks(ax.get_xticks()[::n])
    ax.set_yticks([0,1])
    ax.set_yticklabels(["Away","At Desk"])
    fig.autofmt_xdate()

    ax.text(0.1,0.8,"Total time on {}: {}".format(today,round(total_time_mins,1)),transform = ax.transAxes)

    plt.savefig(graph_file)
    plt.show(block = True)
    pass
atexit.register(save_data)

#Opening the camera
cap = cv2.VideoCapture(0)

#FIND FRAME RATE (17 FPS)
#-----------------------------
#test_time = time.time() + 10
#x = 0
#while(time.time() < test_time):
#    ret,frame = cap.read()
#    x = x+1
#x = x/10
#print(x)
#-----------------------------

#Setting up the DNN
net = cv2.dnn.readNetFromCaffe("dnn_files/deploy.prototxt.txt", "dnn_files/res10_300x300_ssd_iter_140000.caffemodel")

#Using DNN
while(True):
    time_end = time.time() + TIME_INTERVAL
    while(time.time() < time_end):
        if(frames < NUM_FRAMES):
            ret, frame = cap.read()
            #pre-processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (13,13), 0)
            #finding faces in the frame and displaying them
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            numDetections = 0
            #Boxing the faces
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > 0.5:
                    numDetections = numDetections + 1
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    # draw the bounding box of the face along with the associated
                    # probability
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.putText(frame, "You have been sitting here for: {0:.2f} minutes".format(sum(to_graph.values())*TIME_INTERVAL/60), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            #collecting data to give a likelihood of face existence
            frames += 1
            if(numDetections != 0): frames_with_face += 1
            pass
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '): #Why do I need this line, and time.sleep(0.001) doesn't work???
            initEnd = True
            break
        pass
    if(frames != 0 and frames_with_face / frames > MIN_CONF): 
        timeAtDesk += (TIME_INTERVAL)
        to_graph[tuple(time.localtime())] = True
    else:
        to_graph[tuple(time.localtime())] = False
    frames = 0
    frames_with_face = 0
    if(initEnd):
        break

cv2.destroyAllWindows()
cap.release()

#Code that uses a Haar Cascade for Face Detection. This works right now!
while(False):
    time_end = time.time() + TIME_INTERVAL
    while(time.time() < time_end):
        if(frames < NUM_FRAMES):
            ret, frame = cap.read()
            #pre-processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (13,13), 0)
            #finding faces in the frame and displaying them
            faces = frontal_face_default.detectMultiScale(gray, 1.1, 2)
            for (x,y,w,h) in faces:
                frame = cv2.rectangle(frame, (x,y), (x + w,y + h), (0,0,255),2)
                pass
            cv2.putText(frame, "You have been sitting here for: {0:.2f} minutes".format(sum(to_graph.values())*TIME_INTERVAL/60), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (90,50,190), 2)
            #collecting data to give a likelihood of face existence
            frames += 1
            if(len(faces) != 0): frames_with_face += 1
            pass
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '): #Why do I need this line, and time.sleep(0.001) doesn't work???
            initEnd = True
            break
        pass
    if(frames != 0 and frames_with_face / frames > MIN_CONF): 
        timeAtDesk += (TIME_INTERVAL)
        to_graph[tuple(time.localtime())] = True
    else:
        to_graph[tuple(time.localtime())] = False
    frames = 0
    frames_with_face = 0
    if(initEnd):
        break


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
