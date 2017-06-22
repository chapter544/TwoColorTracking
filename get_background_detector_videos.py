import numpy as np
import cv2
import os
from ReadCyanSystemsVideo import GetSequenceNames
from ReadCyanSystemsVideo import ReadCyanSystemsTwoColorVideo
from ReadCyanSystemsVideo import ScaleUint16To255
from WriteBackgroundDetectorVideo import WriteDetectionVideo

#sequenceName = '/data1/TwoColorTracking/sequences/brwncamp5.img'
sequenceDir = '/data1/TwoColorTracking/sequences'
sequences = GetSequenceNames(sequenceDir)

for sequenceName in sequences:
   sequenceFullPath = os.path.join(sequenceDir, sequenceName)
   WriteDetectionVideo(sequenceFullPath)
