import cv2 as cv
from facial_age_estimation_module import FacialAgeEstimator

# init model
FAE = FacialAgeEstimator()
    
# get capture video from camera
live = cv.VideoCapture(0)

while True:
    ret, frame = live.read()
    # check that it's getting vid or not
    if ret == True:            
        # display detected face
        cv.imshow('facial age estimation', FAE.facial_age_estimate(frame))
        if cv.waitKey(1) == ord('q'):
            break
        
live.release()
cv.destroyAllWindows()