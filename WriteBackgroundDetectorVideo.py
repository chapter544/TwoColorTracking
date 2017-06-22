import numpy as np
import cv2
import os
from ReadCyanSystemsVideo import ReadCyanSystemsTwoColorVideo
from ReadCyanSystemsVideo import BackgroundSubtraction
from ReadCyanSystemsVideo import ScaleUint16To255


def WriteDetectionVideo(sequenceName):
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
	varThreshold1 = 5.0
	fgbg1 = cv2.createBackgroundSubtractorKNN(numHistoryFrames, varThreshold1, False)
	varThreshold2 = 3.0
	fgbg2 = cv2.createBackgroundSubtractorKNN(numHistoryFrames, varThreshold2, False)


	# minValidArea
	minValidArea = 10; 


	# morphology close structure element
	kernel = np.ones((5,5), np.uint8)
	smoothingKernel = np.ones((3,3), np.float32)/9.


	# Write to videos
	sequenceShortName = os.path.splitext(os.path.basename(sequenceName))[0]
	if not os.path.exists(sequenceShortName):
		os.mkdir(sequenceShortName)


	#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	#videoObj = cv2.VideoWriter(aviFileName, fourcc, 24., (lw.shape[1], lw.shape[2]*2))

	frameCounter = 0
	while True:
		frameCounter = frameCounter + 1
		print frameCounter
		if frameCounter < numHistoryFrames:
			continue
		elif frameCounter >= sequenceLen:
			break

		lw_frame = ScaleUint16To255(lw[frameCounter], lw_max, lw_min)
		mw_frame = ScaleUint16To255(mw[frameCounter], mw_max, mw_min)
		lw_fg = fgbg1.apply(lw_frame)
		mw_fg = fgbg2.apply(mw_frame)

		upper_frame = np.concatenate( (lw_frame, mw_frame), axis=1)
		lower_frame = np.concatenate( (lw_fg, mw_fg), axis=1)
		frame = np.concatenate( (upper_frame, lower_frame), axis=0 )

		rgb_frame = np.repeat(frame, 3, axis=1)
		rgb_frame = rgb_frame.reshape(frame.shape[0], frame.shape[1], 3)
		fileName = "%s/%05d.jpg" % (sequenceShortName,frameCounter)
		cv2.imwrite(fileName, rgb_frame)
		
	#videoObj.release()



	# Write frames into MP4 
	fps = 24
	outputSequenceFileName = sequenceShortName + '.mp4'

	if os.path.isfile(sequenceShortName):
		os.remove(outputSequenceFileName)

	cmd = ('ffmpeg -start_number ' + str(numHistoryFrames) + 
		   ' -i ' + sequenceShortName + 
		   '/%05d.jpg -c:v libx264 -r ' + str(fps) + ' -pix_fmt yuv420p ' + 
		   outputSequenceFileName)
	os.system( cmd )
