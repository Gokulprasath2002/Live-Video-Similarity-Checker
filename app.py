# Libraries to run this similarity check application
import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# This is the actual function which check for the similarity between the 2 inputs passed which is
# original input and the live video feed input. The below used algorithm is Histogram based sim check
# where an input image has 3 channels RGB and each channel is separated into separate bins and checked for similarity
def sim_checker(actual,present):
    hist_img1 = cv2.calcHist([actual], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_img1[255, 255, 255] = 0
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    hist_img2 = cv2.calcHist([present], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_img2[255, 255, 255] = 0 
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    
    return round(metric_val,1)


# The below function initiates a image segmentation package, which segregates the foreground from the background
# Then the input size is normalized by resizing it
def image_preprocessor(actual,present):
    segmentor = SelfiSegmentation()

    actual_resized = cv2.resize(actual,(300,300))
    present_resized = cv2.resize(present,(300,300))

    image1 = segmentor.removeBG(actual_resized)
    image2 = segmentor.removeBG(present_resized)
    #cv2.imshow('frame', image2)

    return sim_checker(image1,image2)

# The below function load the 2 inputs, where keeping the actual input's channel RGB as it is, without any changes
def image_loader(actual,present):
    image1 = cv2.imread(actual,cv2.IMREAD_UNCHANGED)

    return image_preprocessor(image1,present)

# The below function checks whether any human faces inside the video feed or not by returning the confidence score
def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]
    return confidence        

# This code is used to start the video capture, where 0 indicates the video feed from laptop camera
vid = cv2.VideoCapture(0) 
  
while(True): 

    ret, frame = vid.read() 

    # Pass in your actual image, where you would checking similarity with
    score = image_loader("data\PassportPhotograph - Gokulprasath.jpeg",frame)
    print(score)
    # Only if the face detection confidence is above 0.5, similarity check starts
    if(detect_face(frame)>=0.5):
        if score==-0.0:
            text = "Not Similar"
            cv2.putText(frame, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            cv2.imshow('frame', frame)
        elif score<=-0.0:
            text = "Not Similar"
            cv2.putText(frame, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            cv2.imshow('frame', frame)
        elif score==0.0:
            text = "Similar"
            cv2.putText(frame, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            cv2.imshow('frame', frame)
        elif score>=0.0:
            text = "Similar"
            cv2.putText(frame, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            cv2.imshow('frame', frame)
        else:
            text = "Similar"
            cv2.putText(frame, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            cv2.imshow('frame', frame)
    else:
        text = "No faces found"
        cv2.putText(frame, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        cv2.imshow('frame', frame)

    # Once user clicks q, the program ends
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    
vid.release() 

cv2.destroyAllWindows()  
