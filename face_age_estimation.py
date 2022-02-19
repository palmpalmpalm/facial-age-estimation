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
    
    # draw rectangles
    for (x,y,w,h) in bboxes:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    # set padding to 20
    padding = 20
    
    for box in bboxes:
        # get face frame
        face = frame[max(0, box[1] - padding):min(box[1] + box[3] + padding, frame.shape[0] - 1),
               max(0, box[0] - padding):min(box[0] + box[2] + padding, frame.shape[1] - 1)]
        
        preprocessed_face = preprocess_for_fae(face)
        
        age = facial_age_estimation_model.predict(np.array([preprocessed_face]))
        print(age)
        
        
    return frame


def main():
    live = cv.VideoCapture(0)
    while True:
        ret, frame = live.read()
        if ret == True:
            
            # display detected face
            cv.imshow('live face detection', facial_age_estimate(frame))
            if cv.waitKey(1) == ord('q'):
                break
            
    live.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
