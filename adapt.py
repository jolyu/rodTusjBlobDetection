import cv2

th = cv2.imread('Bilder/Birds.jpg',0)
th = cv2.medianBlur(th,5)
th = cv2.resize(th, (650,500))
ret,th1 = cv2.threshold(th,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(th1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow("2",th2)
key = cv2.waitKey()
while key != 27: # exit on ESC
    key = cv2.waitKey()
cv2.destroyAllWindows()