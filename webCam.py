import cv2
import blobDetect as bd
import numpy as np

redLower = (0,10,10)                                                #100,130,50 skal være for å filtrere rød farge
redUpper = (80,255,255)                                             #200,200,130

def adaptive_thresh(img,thresh,max,type):
   
    if(type == "bin"):
        ret,th1 = cv2.threshold(img,thresh,max,cv2.THRESH_BINARY) 
        return th1
    elif(type == "mean"):
        img_blur = cv2.medianBlur(img,5)
        ret,th1 = cv2.threshold(img_blur,thresh,max,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(th1,max,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        return th2
    else:
        img_blur = cv2.meadianBlur(img,5)
        ret,th1 = cv2.threshold(img_blur,thresh,max,cv2.THRESH_BINARY)
        th3 = cv2.adaptiveThreshold(th1,max,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        return th3 

if __name__ == "__main__":
    cv2.namedWindow("preview")                                      #åpner vindu
    vc = cv2.VideoCapture(0)                                        #tar er bilde med webCam

    if vc.isOpened():                                               # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    frame = cv2.resize(frame, (650,500))                            #fungerer ikke, burde skalerer bedre etter oppløsning til kamera
    cv2.imshow("preview", frame)                                    #viser frame
    detector = bd.blobDetector()                                    #første detection


    while rval:                                                    #fortsetter å ta bilder så lenge det går, methink
        
        rval, frame = vc.read()                                     #tar nytt bilde
        frame = cv2.resize(frame, (650,500))                        #reskalerer for vindu, fungerer ikke som planlagt (endre pixler)                      
        mask = cv2.inRange(frame, redLower, redUpper)               #har ingen anelse hvordan mask opplegget fungerer, men det må til for å filterer bort alt fra bildet
        mask = cv2.erode(mask, None, iterations=0)                  #som ikke er rødt
        mask = cv2.dilate(mask, None, iterations=0) 
        frame = cv2.bitwise_and(frame,frame,mask = mask)            #dette gir et rart bilde
        cv2.imshow("34", frame)
       
        #Adaptive thresh forsøk
        th2 = adaptive_thresh(frame,127,255,"mean")
        th1 = adaptive_thresh(frame,127,255,"gauss")
        th = adaptive_thresh(frame,127,255,"bin")
        cv2.imshow("mean", th2)
        cv2.imshow("gauss", th1)
        cv2.imshow("bin", th)

        #Adaptive thresh forsøk 
        (thres, frame) = cv2.threshold(frame, 190, 255, cv2.THRESH_BINARY)  #men med å gjøre dette får man svart-rødt bilde
        cv2.imshow("1", frame)
        frame = 255 -frame                                          #her inverteres svart rødt bilde (detector-funksjonen detekterer mørke objekt på lys bakgrunn, dette kan endres though)
        cv2.imshow("2", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)             #gjør bildet om til gråskala
        cv2.imshow("3", frame)
        (thres, frame) = cv2.threshold(frame, 190, 255, cv2.THRESH_BINARY)  #her bruker man en treshold så man får bilder i svart-hvitt
        cv2.imshow("4", frame)
        newFrame = bd.detectStuff(frame, detector)                  #bildegjennkjenning
        cv2.imshow("preview", newFrame)                             #viser resultatet og starter loop på nytt

    
        if cv2.waitKey(30) == 27: # exit on ESC                     #avslutter programmet og lukker alle viduer dersom man trykker ESC
            break
    cv2.destroyWindow("preview")
