import numpy as np
from ReadCyanSystemsVideo import ScaleUint16To255
import cv2


def BuildBackgroundModels(lw, mw, numHistoryFrames=100, \
                          varThreshold1=5.0, varThreshold2=3.0, \
                          detectShadows=False, \
                          gamma_lw=.9, gamma_mw=0.05):
    lw_max = np.amax(lw)
    mw_max = np.amax(mw)
    lw_min = np.amin(lw)
    mw_min = np.amin(mw)
    lw_range = np.amax(lw) - np.amin(lw)
    mw_range = np.amax(mw) - np.amin(mw)

    # create background subtraction model
    fgbg1 = cv2.createBackgroundSubtractorKNN(numHistoryFrames, varThreshold1, detectShadows)
    fgbg2 = cv2.createBackgroundSubtractorKNN(numHistoryFrames, varThreshold2, detectShadows)
    frameCounter = 0
    while True:
        frameCounter = frameCounter + 1
        if frameCounter > numHistoryFrames:
            break

        lw_frame = ScaleUint16To255(lw[frameCounter], lw_max, lw_min, gamma_lw)
        mw_frame = ScaleUint16To255(mw[frameCounter], mw_max, mw_min, gamma_mw)
        lw_fg = fgbg1.apply(lw_frame)
        mw_fg = fgbg2.apply(mw_frame)

    return fgbg1, fgbg2




def GetTargetObservation(target_cts, pred_x):
    if not target_cts:
        return pred_x

    obj_features = np.zeros( (4, len(target_cts)) )
    for idx, c in enumerate(target_cts):
        # bounding box with minimum area, so it consider rotation.
        # minAreaRect returns a Box2D structure 
		# (topleftcorner(x,y), (width, height), angle)
        rect = cv2.minAreaRect(c)
        #print rect
        top_corner = rect[0]
        obj_dim = rect[1]
        #print top_corner
        #print obj_dim
        obj_features[:,idx] = np.array([top_corner[0], \
                                        top_corner[1] , \
                                        obj_dim[0], obj_dim[1]])
        #print top_corner, obj_dim 

    #print("Feature obj")
    #print obj_features
    #dist_xy_list = []
    #dist_wh_list = []
    return obj_features





def GetTargetCorrespondent(obj_features, pred_x, xy_dist_max=50.,
        wh_dist_max=20.):
    DISTANCE_TYPES = 2 # xy and wh

    #print obj_features
    dist_lw = np.zeros((DISTANCE_TYPES,obj_features.shape[1]))
    for idx in range(obj_features.shape[1]):
        temp_features = obj_features[:,idx]
        dist_lw[0,idx]= abs(pred_x[0] - temp_features[0]) + \
                  abs(pred_x[2] - temp_features[1])
        dist_lw[1,idx] = abs(pred_x[4] - temp_features[2]) + \
                  abs(pred_x[5] - temp_features[3])

    #print dist_lw
    xy_min_idx = np.argmin(dist_lw[0,:])
    wh_min_idx = np.argmin(dist_lw[1,:])
    #print('xy index %d' % xy_min_idx)
    #print('wh ndex %d' % wh_min_idx)

    is_valid = True
    if dist_lw[0,xy_min_idx] > xy_dist_max or \
            dist_lw[1,wh_min_idx] > wh_dist_max:
        is_valid = False

    xy_err = dist_lw[0, xy_min_idx]
    return obj_features[:,xy_min_idx], is_valid, xy_err 
    
	


def GetTargetCorrespondentTwoColor(lw_features, mw_features, pred_x):
    lw_feature, lw_valid, lw_err = GetTargetCorrespondent(lw_features, pred_x,
            xy_dist_max=30.,
            wh_dist_max=20.)
    mw_feature, mw_valid, mw_err = GetTargetCorrespondent(mw_features, pred_x,
            xy_dist_max=30.,
            wh_dist_max=20.)

    z_valid = not (lw_valid and mw_valid)
    selected_feature = []

    if z_valid:
        selected_feature = lw_feature
        if lw_err > mw_err:
            selected_feature = mw_feature

    #if lw_valid and not mw_valid:
    #    z_valid = True
    #    selected_feature = lw_feature
    
    #if not mw_valid and mw_valid:
    #    z_valid = True
    #    selected_feature = mw_feature
    return selected_feature, z_valid 
	   

def BackgroundSubtraction(frame, fgbg, kernel, validRegionArea):
    fgmask = fgbg.apply(frame)
    
    _, thresh = cv2.threshold(fgmask, 127, 255, 0)
    fgmask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    #cv2.imwrite("test.jpg", fgmask)

    ret, cts, hier = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cts_sorted = sorted(cts, key = cv2.contourArea, reverse=True)

    good_cts = []
    for c in cts_sorted:
        area = cv2.contourArea(c)
        if area > validRegionArea:
            good_cts.append(c)

    return good_cts



def GetContoursFromBackgroundMask(fgmask, validRegionArea):
    ret, cts, hier = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cts_sorted = sorted(cts, key = cv2.contourArea, reverse=True)

    good_cts = []
    for c in cts_sorted:
        area = cv2.contourArea(c)
        if area > validRegionArea:
            good_cts.append(c)

    return good_cts



def GetBackgroundSubtraction(frame, fgbg, kernel):
    fgmask = fgbg.apply(frame)
    
    _, thresh = cv2.threshold(fgmask, 127, 255, 0)
    fgmask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    return fgmask
