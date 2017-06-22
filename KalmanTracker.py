import numpy as np
import cv2
import os
import ReadCyanSystemsVideo as cyanSystem
import TrackingUtility as tracking 
from KalmanFilter import KalmanFilter

#sequence_list = cyanSystem.GetSequenceList()
#sequence_list = cyanSystem.GetRandomSequence()
sequence_list = cyanSystem.GetSequence(2)
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

varThresh1=32.0
varThresh2=32.0

gamma_lw=0.9
gamma_mw=0.01

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
    lw_bg, mw_bg = tracking.BuildBackgroundModels(lw, mw, historyFrames, \
                                                  varThresh1, varThresh2, \
                                                  False, gamma_lw, gamma_mw) 
    xhat_lw = np.zeros( (6, sequence['end_frame']-sequence['start_frame']+2) )
    xhat_mw = np.zeros( (6, sequence['end_frame']-sequence['start_frame']+2) )
    xhat_all = np.zeros( (6, sequence['end_frame']-sequence['start_frame']+2) )
    xhat_lw[:,0] = [sequence['x_center'], 0.1, sequence['y_center'], 0.1, \
                    sequence['half_width']*2+1, sequence['half_height']*2+1]
    Ppost_lw = np.zeros(F.shape)
    xpred_lw = xhat_lw[:,0]

    xhat_mw[:,0] = xhat_lw[:,0]
    Ppost_mw = np.zeros(F.shape)
    xpred_mw = xhat_mw[:,0]

    xhat_all[:,0] = xhat_lw[:,0]
    Ppost_all = np.zeros(F.shape)
    xpred_all = xhat_all[:,0]

    kalman = KalmanFilter(F, H, Q, R)


    frameCounter = historyFrames
    idx = 0
    while True:
        frameCounter = frameCounter + 1
        idx = idx + 1 
        #print frameCounter
        if frameCounter > sequenceLen:
            break

        lw_frame = cyanSystem.ScaleUint16To255(lw[frameCounter], \
                lw_max, lw_min, gamma_lw)
        mw_frame = cyanSystem.ScaleUint16To255(mw[frameCounter], \
                mw_max, mw_min, gamma_mw)
        lw_mask = tracking.GetBackgroundSubtraction(lw_frame, \
                lw_bg, morphKernel)
        mw_mask = tracking.GetBackgroundSubtraction(mw_frame, \
                mw_bg, morphKernel)

        lw_cts = tracking.GetContoursFromBackgroundMask(lw_mask.copy(),  \
                minValidRegionArea)
        mw_cts = tracking.GetContoursFromBackgroundMask(mw_mask.copy(), \
                minValidRegionArea)

        #blah = np.asarray(np.dstack((lw_mask, lw_mask, lw_mask)), 
        #dtype=np.uint8)
        #for c in lw_cts:
        #    rect = cv2.minAreaRect(c)
        #    box = np.int0(cv2.boxPoints(rect))
        #    cv2.drawContours(blah, [box], -1, (0, 255, 0), 2)

        #print('Writing lw mask %d' % frameCounter)
        #fileName = "%s/%05d_lw_mask.jpg" % (sequence['ref'], frameCounter)
        #cv2.imwrite(fileName, blah)

        #print('Begin filtering %d' % frameCounter)
        #print('predicted')
        # LW correspondent
        lw_feat = tracking.GetTargetObservation(lw_cts, xpred_lw)
        z_lw, z_lw_valid, err_lw = \
            tracking.GetTargetCorrespondent(lw_feat, xpred_lw, 50., 20.)

        # MW correspondent
        mw_feat = tracking.GetTargetObservation(mw_cts, xpred_mw)
        z_mw, z_mw_valid, error_mw = \
            tracking.GetTargetCorrespondent(mw_feat, xpred_mw, 50., 20.)

        # Two color
        all_feat_lw = tracking.GetTargetObservation(lw_cts, xpred_all)
        all_feat_mw = tracking.GetTargetObservation(mw_cts, xpred_all)
        z_all, z_all_valid = \
            tracking.GetTargetCorrespondentTwoColor(lw_feat, mw_feat, xpred_all)

        if not z_all_valid:
            print('Coasting %d, %d' % (frameCounter, z_all_valid))

        #print z_lw_valid, z_mw_valid, z_valid
        #print lw_z
        #print xpred_all
        #print mw_z
        #print xpred_all
        #print("Doing %d" % frameCounter)
        #print("lw valid: %d" % (z_lw_valid)) 
        #print z_lw
        #print("mw valid: %d" % (z_mw_valid)) 
        #print z_mw
        #print("all valid: %d" % (z_all_valid)) 
        #print z_all

        #if not z_lw_valid and not z_mw_valid:
        #    print "Warning: invalid Z"
        #print('z')
        #print lw_z, mw_z

        #break

        #print xpred
        #print Ppost
        #print lw_z
        xhat_lw[:,idx], Ppost_lw = \
                kalman.filter(xpred_lw, Ppost_lw, z_lw, z_lw_valid)
        xpred_lw = xhat_lw[:,idx]

        xhat_mw[:,idx], Ppost_mw = \
                kalman.filter(xpred_mw, Ppost_mw, z_mw, z_mw_valid)
        xpred_mw = xhat_mw[:,idx]

        xhat_all[:,idx], Ppost_all = \
                kalman.filter(xpred_all, Ppost_all, z_all, z_all_valid)
        xpred_all = xhat_all[:,idx]
        #print xhat_lw[:,idx], xhat_mw[:,idx]
        
        #rect = ((xpred[0] - xpred[4]*0.5, xpred[2] - xpred[5]*0.5), (xpred[4], xpred[5]), 0.0)
        rect = ((xpred_all[0], xpred_all[2]), (xpred_all[4], xpred_all[5]), 0.0)
        box = np.int0(cv2.boxPoints(rect))

        rect_lw = ((xpred_lw[0], xpred_lw[2]), (xpred_lw[4], xpred_lw[5]), 0.0)
        box_lw = np.int0(cv2.boxPoints(rect_lw))

        rect_mw = ((xpred_mw[0], xpred_mw[2]), (xpred_mw[4], xpred_mw[5]), 0.0)
        box_mw = np.int0(cv2.boxPoints(rect_mw))


        lw_rgb = cyanSystem.to_rgb(lw_frame)
        #print lw_rgb.shape, lw_rgb.dtype
        cv2.drawContours(lw_rgb, [box], -1, (255, 0, 0), 1)
        cv2.drawContours(lw_rgb, [box_lw], -1, (0, 255, 0), 1)
        cv2.drawContours(lw_rgb, [box_mw], -1, (0, 0, 255), 1)

        mw_rgb = cyanSystem.to_rgb(mw_frame)
        cv2.drawContours(mw_rgb, [box], -1, (255, 0, 0), 1)
        cv2.drawContours(mw_rgb, [box_lw], -1, (0, 255, 0), 1)
        cv2.drawContours(mw_rgb, [box_mw], -1, (0, 0, 255), 1)

        lw_mask_rgb = cyanSystem.to_rgb(lw_mask)
        mw_mask_rgb = cyanSystem.to_rgb(mw_mask)
        upper_frame = np.concatenate( (lw_rgb, mw_rgb), axis=1)
        lower_frame = np.concatenate( (lw_mask_rgb, mw_mask_rgb), axis=1)

        out_frame = np.concatenate( (upper_frame, lower_frame), axis=0 )

        #rgb_frame = np.repeat(frame, 3, axis=1)
        #rgb_frame = rgb_frame.reshape(frame.shape[0], frame.shape[1], 3)

        fileName = "%s/%05d.jpg" % (sequence['ref'], frameCounter)
        cv2.imwrite(fileName, lw_frame)
        cv2.imshow('frame', out_frame)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cv2.destroyAllWindows()
