import numpy as np
import cv2
import os
import ReadCyanSystemsVideo as cyanSystem
import TrackingUtility as tracking 

#sequence_list = cyanSystem.GetSequenceList()
#sequence_list = cyanSystem.GetRandomSequence()
sequence_list = cyanSystem.GetSequence(3)
sequenceDir = '/data1/TwoColorTracking/sequences'

morphKernel = np.ones((5,5), np.uint8)
smoothingKernel = np.ones((3,3), np.float32)/9.
minValidRegionArea = 10


# Assume the constant velocity model
delta_t = 1;
F = np.array([[1, delta_t, 0, 0, 0, 0], 
              [0, 1, 0, 0, 0, 0], 
              [0, 0, 1, delta_t, 0, 0], 
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])
# noise variance
Q = .5
R = 2.0

varThresh1=10.0
varThresh2=8.0

print sequence_list

for sequence in sequence_list:
    sequenceName = sequenceDir + '/' + sequence['name']
    # create sequence directory 
    if not os.path.exists(str(sequence['ref'])):
        os.makedirs(str(sequence['ref']))

    lw, mw = cyanSystem.ReadCyanSystemsTwoColorVideo(sequenceName, crop=True)
    lw_max = np.amax(lw)
    mw_max = np.amax(mw)
    lw_min = np.amin(lw)
    mw_min = np.amin(mw)
    lw_range = np.amax(lw) - np.amin(lw)
    mw_range = np.amax(mw) - np.amin(mw)

    sequenceLen = sequence['end_frame']

    historyFrames = sequence['start_frame']-1
    lw_bg, mw_bg = tracking.BuildBackgroundModels(lw, mw, historyFrames, varThresh1, varThresh2)
    
    xhat = np.zeros( (6, sequence['end_frame']-sequence['start_frame']+2) )
    #c, v1, r, v2, w, h = (sequence['x_center'], 0.1, sequence['y_center'], 0.1, \
    #                      sequence['half_width']*2+1, sequence['half_height']*2+1)
    #xhat[:,0] = [sequence['x_center'], 0.1, sequence['y_center'], 0.1, \
    #             sequence['half_width']*2+1, sequence['half_height']*2+1]
    
    c, r, w, h = (sequence['x_center']-sequence['half_width'], \
                          sequence['y_center']-sequence['half_height'], \
                          sequence['half_width']*2+1, sequence['half_height']*2+1)

    xhat[:,0] = (c, 0, r, 0, w, h)


    #xpred = xhat[:,0]
    track_window = (c, r, w, h)
    print track_window
    
    lw_frame = cyanSystem.ScaleUint16To255(lw[historyFrames], lw_max, lw_min)
    mw_frame = cyanSystem.ScaleUint16To255(mw[historyFrames], mw_max, mw_min)

    #mask = np.zeros(lw_frame.shape, dtype=np.uint8)
    #mask[r:r+h,c:c+w] = 1
    #cv2.imwrite("blah.jpg", lw_frame*mask)
    roi_hist = cv2.calcHist([lw_frame[r:r+h,c:c+w]], [0], None, [64], [0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1 )

    frameCounter = historyFrames
    idx = 0
    while True:
        frameCounter = frameCounter + 1
        idx = idx + 1 
        #print frameCounter
        if frameCounter > sequenceLen:
            break

        lw_frame = cyanSystem.ScaleUint16To255(lw[frameCounter], lw_max, lw_min)
        mw_frame = cyanSystem.ScaleUint16To255(mw[frameCounter], mw_max, mw_min)

        dst = cv2.calcBackProject([lw_frame], [0], roi_hist, [0, 256], 1)
        fileName = "%s/%05d.jpg" % (sequence['ref'], frameCounter)
        cv2.imwrite(fileName, dst)

        rect, track_window = cv2.meanShift(dst, track_window, term_crit)
        #rect, track_window = cv2.CamShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        print track_window
        img2 = cv2.rectangle(lw_frame, (x,y), (x+w, y+h), 255, 2)

        #print box
        #fileName = "%s/%05d.jpg" % (sequence['ref'], frameCounter)
        #cv2.imwrite(fileName, img2)
        cv2.imshow('frame', img2)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cv2.destroyAllWindows()
