from deepface import DeepFace
import cv2
from time import sleep

cap=cv2.VideoCapture(0)
facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    success,img=cap.read()
    imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facecascade.detectMultiScale(imggray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        face_roi=img[y:y+h,x:x+w]
        face_roi=cv2.resize(face_roi,(48,48))

        try:
            res=DeepFace.analyze(face_roi,actions=['emotion'],enforce_detection=False)

            
            cv2.putText(img,str(res[0]['dominant_emotion']),(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            
                 
        except:
            pass
        



    
    cv2.imshow('Webcam',img)
   
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

