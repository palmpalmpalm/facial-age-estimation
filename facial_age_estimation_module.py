import cv2 as cv
from tensorflow import keras
import numpy as np




class FacialAgeEstimator:
    
    def __init__(self):
        # load models
        self.face_detection_model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.facial_age_estimation_model = keras.models.load_model('models/model1')
    
    # return bboxes for faces by using face_detection_model
    def get_face_box(self, frame):
        
        # preprocessing
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # inferencing
        detections = self.face_detection_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
            
        return detections

    # preprocesing function for facial_age_estimation_model
    def preprocess_for_fae(self, img):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        width = 100
        height = 100
        resize_img = cv.resize(rgb_img, (width,height), interpolation=cv.INTER_AREA)
        return np.array(resize_img)/255.0

    def facial_age_estimate(self, frame):
        
        # get face bboxess from face_detection_model
        bboxes = self.get_face_box(frame)
        
        # set padding to 20
        padding = 20
        
        # inferencing
        for box in bboxes:
            # box coordinate
            x,y,w,h = box[0],box[1],box[2],box[3] 
            
            # get face frame
            face = frame[max(0, y - padding):min(y + h + padding, frame.shape[0] - 1),
                max(0, x - padding):min(x + w + padding, frame.shape[1] - 1)]
            
            preprocessed_face = self.preprocess_for_fae(face)
            
            age = self.facial_age_estimation_model.predict(np.array([preprocessed_face]))
            
            # draw result on frame
            label = 'Age: {}'.format(int(age[0][0]))
            cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)            
            

        return frame
    
    def get_age_by_face_image(self, img):              
        
        # preprocessing image
        preprocessed_face = self.preprocess_for_fae(img)
        
        # inferencing
        age = self.facial_age_estimation_model.predict(np.array([preprocessed_face]))

        return int(age[0][0]) 
        
    


def main():
    
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


if __name__ == '__main__':
    main()
