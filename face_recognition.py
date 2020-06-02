import cv2
import numpy as np
cascadePath = "haarcascade_frontalface_default.xml"
cam = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cascadePath);
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

id=0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im =cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3,5)


    for(x,y,w,h) in faces:

            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 2)


            Id,Name = recognizer.predict(gray[y:y+h,x:x+w])
            cv2.putText(im,str(Id),(x,y+h),font,2,(255,0,0),2,cv2.LINE_AA)
            print("Authenticated")
            cv2.imshow("face",im)
            cv2.waitKey(1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
