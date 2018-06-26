import cv2

def faceRecognitionInImage():
    
    image=cv2.imread('/home/revanth/Downloads/exampleForFaceRecognition.jpg')
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faceCascade=cv2.CascadeClassifier("/home/revanth/bytemInternship/faceDetectionUsingOpenCV/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray, 1.1, 5)
    for x,y,w,h in faces :
        cv2.rectangle(image, (x,y), (x+w,y+h),(0,255,0),2)
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def faceRecognitionInWebcam(): 
 
    face_cascade = cv2.CascadeClassifier('/home/revanth/bytemInternship/faceDetectionUsingOpenCV/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/home/revanth/bytemInternship/faceDetectionUsingOpenCV/opencv/data/haarcascades/haarcascade_eye.xml') 
    cap = cv2.VideoCapture(0)
    while 1: 
        ret, img = cap.read() 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray) 
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows() 




if __name__ == "__main__":
    
   
    print "Enter \n1 for Face Recognition in  Image \n2 for Face Recognition in Webcam"
    user=input()
    if user==2 :
        face1=faceRecognitionInWebcam()
    elif user==1:
        face1=faceRecognitionInImage()
    else:
        print "wrong input"
    
    
    #for more comments and explanation go to sourceForFaceRecognition.py	 in the same directory
