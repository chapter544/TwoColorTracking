import numpy as np
import cv2
from ReadCyanSystemsVideo import ReadCyanSystemsTwoColorVideo
from ReadCyanSystemsVideo import BackgroundSubtraction
from ReadCyanSystemsVideo import ScaleUint16To255


#sequenceName = '/data1/TwoColorTracking/sequences/brwncamp1.img'
#sequenceName = '/data1/TwoColorTracking/sequences/brwncamp3.img'
sequenceName = '/data1/TwoColorTracking/sequences/brwncamp8.img'
#sequenceName = '/data1/TwoColorTracking/sequences/brwncamp5.img'
#sequenceName = '/data1/TwoColorTracking/sequences/vons6.img'
#sequenceName = '/data1/TwoColorTracking/sequences/vons1.img'
#cap = cv2.VideoCapture('/data1/TwoColorTracking/sequences/avi/brwncamp1_video.avi')

lw, mw = ReadCyanSystemsTwoColorVideo(sequenceName, crop=True)

print np.amax(lw), np.amin(lw)
print np.amax(mw), np.amin(mw)
lw_max = np.amax(lw)
mw_max = np.amax(mw)
lw_min = np.amin(lw)
mw_min = np.amin(mw)
lw_range = np.amax(lw) - np.amin(lw)
mw_range = np.amax(mw) - np.amin(mw)
#if lw_range > mw_range:
#    max_val = lw_max
#else:
#    max_val = mw_max

sequenceLen = lw.shape[0]

# create background subtraction model
numHistoryFrames = 100
#varThreshold1 = 10.0
#fgbg1 = cv2.createBackgroundSubtractorMOG2(numHistoryFrames, varThreshold1, False)
#varThreshold2 = 5.0
#fgbg2 = cv2.createBackgroundSubtractorMOG2(numHistoryFrames, varThreshold2, False)



varThreshold1 = 5.0
fgbg1 = cv2.createBackgroundSubtractorKNN(numHistoryFrames, varThreshold1, False)
varThreshold2 = 3.0
fgbg2 = cv2.createBackgroundSubtractorKNN(numHistoryFrames, varThreshold2, False)



# minValidArea
minValidArea = 10; 


# morphology close structure element
kernel = np.ones((5,5), np.uint8)
smoothingKernel = np.ones((3,3), np.float32)/9.

frameCounter = 0
while True:
    frameCounter = frameCounter + 1
    if frameCounter < numHistoryFrames:
        continue
    elif frameCounter >= sequenceLen:
        break

    #lw_frame = lw[frameCounter]
    #mw_frame = mw[frameCounter]
    lw_frame = ScaleUint16To255(lw[frameCounter], lw_max, lw_min)
    mw_frame = ScaleUint16To255(mw[frameCounter], mw_max, mw_min)
    #print np.amax(lw_frame), np.amin(lw_frame), np.amax(mw_frame), np.amin(mw_frame)
    #lw_frame = cv2.equalizeHist(lw_frame)
    #mw_frame = cv2.equalizeHist(mw_frame)
    #lw_frame = cv2.filter2D(lw_frame, -1, smoothingKernel)
    #mw_frame = cv2.filter2D(mw_frame, -1, smoothingKernel)
    #mw_frame = cv2.medianBlur(mw_frame, 3)
    
    lw_cts = BackgroundSubtraction(lw_frame, fgbg1, kernel, minValidArea)
    mw_cts = BackgroundSubtraction(mw_frame, fgbg2, kernel, minValidArea)

    for c in lw_cts:
	    # bounding box with minimum area, so it consider rotation.
        # minAreaRect returns a Box2D structure (topleftcorner(x,y), (width, height), angle)
        rect = cv2.minAreaRect(c)

        # to draw rectangle, we need 4 corners, so we need to call boxPoints
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(lw_frame, [box], -1, (0, 255, 0), 2)


    for c in mw_cts:
	    # bounding box with minimum area, so it consider rotation.
        # minAreaRect returns a Box2D structure (topleftcorner(x,y), (width, height), angle)
        rect = cv2.minAreaRect(c)

        # to draw rectangle, we need 4 corners, so we need to call boxPoints
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(mw_frame, [box], -1, (0, 255, 0), 2)


    frame = np.concatenate( (lw_frame, mw_frame), axis=1)
    cv2.imshow('frame', frame)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#cap.release()

cv2.destroyAllWindows()

