# Desk Face Detection
A program to measure time spent working by OpenCV face recognition. Future goals include recording eye direction to measure focus.

I spend a lot of time at my desk - maybe too much. But how much exactly? My initial idea was a screen tracking software. Plenty already exist, so I just downloaded one and began using it. Pretty soon I realized that I spend a lot of time off the screen, but still at my desk - reading, thinking, spinning around in my chair, so on. Two alternatives were a face-tracking algorithm, and an Arduino-based pressure/proximity sensor on my chair. I went with the face-tracker, mainly because I had no idea how it would work, and I wanted to find out! There's still work to be done, but here's my functioning prototype.
## Use
This program makes use of the OpenCV library, so you'll have to download that before you run the code. Make sure you store the dnn_files folder and Haar Cascade XML in the same directory as the python file. To stop the program, open the tab which shows the annotated webcam output and hold space bar. Your daily stats will be logged as an image file in an auto-generated daily_graphs folder.
## How It Works
After reading up and experimenting with a couple of different face detection algorithms, I went with the OpenCV DNN. It works much better than the Haar Cascade Classifier, but I've included the code and files for both.
