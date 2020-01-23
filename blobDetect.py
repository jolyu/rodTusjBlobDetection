import cv2
import numpy as np

def readImage():
    # read image in greyscale

    img = cv2.imread('Bilder/Birds.jpg')
    img = cv2.resize(img, (650,500))
    #img = img[:,:,1]
    #img = (255-img)
    return img

def blobDetector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 100
    params.maxThreshold = 255


    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01


    #determine cv version
    print(cv2.__version__)
    if cv2.__version__.startswith('2'):
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)
    
    return detector

def detectStuff(img, detector):
    # detect suff
    keypoints = detector.detect(img)

    #draw detected keypoints as red circles
    imgKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]),(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return imgKeyPoints

if __name__ == "__main__":
    img = readImage()
    detect = blobDetector()
    newImg = detectStuff(img, detect)

    #display results
    cv2.imshow("preview", newImg)

    key = cv2.waitKey()
    while key != 27: # exit on ESC
        key = cv2.waitKey()
    
    cv2.destroyAllWindows()