import cv2 as cv
from tensorflow import keras
import numpy as np

# load models
face_detection_model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
facial_age_estimation_model = keras.models.load_model('models/model1')


def get_face_box(frame):
    
    # preprocessing
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # inferencing
    detections = face_detection_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
        
    return detections

def preprocess_for_fae(img):
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    width = 100
    height = 100
    resize_img = cv.resize(rgb_img, (width,height), interpolation=cv.INTER_AREA)
    return np.array(resize_img)/255.0

def facial_age_estimate(frame):
    
    # get face bboxess from face_detection_model
    bboxes = get_face_box(frame)
    
    # set padding to 20
    padding = 20
    
    # inferencing
    for box in bboxes:
        # box coordinate
        x,y,w,h = box[0],box[1],box[2],box[3] 
        
        # get face frame
        face = frame[max(0, y - padding):min(y + h + padding, frame.shape[0] - 1),
               max(0, x - padding):min(x + w + padding, frame.shape[1] - 1)]
        
        preprocessed_face = preprocess_for_fae(face)
        
        age = facial_age_estimation_model.predict(np.array([preprocessed_face]))
        
        # draw result on frame
        label = 'Age: {}'.format(int(age[0][0]))
        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)              
        

    return frame


def main():
    live = cv.VideoCapture(0)
    while True:
        ret, frame = live.read()
        if ret == True:
            
            # display detected face
            cv.imshow('facial age estimation', facial_age_estimate(frame))
            if cv.waitKey(1) == ord('q'):
                break
            
    live.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
