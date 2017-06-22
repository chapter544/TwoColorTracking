import numpy as np
import cv2
import os
from ReadCyanSystemsVideo import GetSequenceNames
from ReadCyanSystemsVideo import ReadCyanSystemsTwoColorVideo

#sequenceName = '/data1/TwoColorTracking/sequences/brwncamp5.img'
sequenceDir = '/data1/TwoColorTracking/sequences'
sequences = GetSequenceNames(sequenceDir)

for sequenceName in sequences:
   sequenceFullPath = os.path.join(sequenceDir, sequenceName)
   lw, mw = ReadCyanSystemsTwoColorVideo(sequenceFullPath, crop=True)
   lw_max = np.amax(lw)
   mw_max = np.amax(mw)
   print('%s' % sequenceName)
   print('Length %d' % lw.shape[0])
   print('LW: %d %d' % (np.amax(lw), np.amin(lw)))
   print('MW: %d %d' % (np.amax(mw), np.amin(mw)))
   print('\n')
