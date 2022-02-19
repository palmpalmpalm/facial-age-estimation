import cv2 as cv


pretrained_model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


def main():
    live = cv.VideoCapture(0)
    while True:
        ret, frame = live.read()
        if ret == True:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            coordinate_list = pretrained_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))

            # draw rectangles
            for (x,y,w,h) in coordinate_list:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
            # display detected face
            cv.imshow('live face detection', frame)
            if cv.waitKey(1) == ord('q'):
                break
    live.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
