import numpy as np
import cv2
from os import listdir
from os.path import isfile, join, splitext


def to_rgb(im):
	#return np.repeat(im.astype(np.uint8), 3, 2)
	return np.dstack([im.astype(np.uint8)] * 3).copy(order='C')




def ReadCyanSystemsTwoColorVideo(videoFileName, crop=False):
    with open(videoFileName, "rb") as f:
        header_bytes = f.read(512)
        frame_bytes = f.read()

    header_info = np.fromstring(header_bytes, dtype=np.uint16)
    width = header_info[1].astype(np.int32)
    height = header_info[2].astype(np.int32)
    raw_frames = np.fromstring(frame_bytes, dtype=np.uint16)
    big_frames = np.reshape(raw_frames, (-1, height,width))

    lw = big_frames[:,:,:width/2]
    mw = big_frames[:,:,width/2:]

    if crop is True:
        lw = lw[:,:,:lw.shape[1]]
        mw = mw[:,:,:mw.shape[1]]
        #print lw.shape, mw.shape

    return lw, mw






def ScaleUint16To255(frame, max_val, min_val, gamma=.7): 
    max_val = max_val ** gamma
    min_val = min_val ** gamma

    temp = frame.astype(np.float32) ** gamma
    temp = np.minimum(temp, max_val)
    # 14-bit precision
    #temp_frame = (temp.astype(np.float32) - min_val) / (max_val - min_val) * 255.
    temp_frame = (temp - min_val) / (max_val - min_val) * 255.
    return temp_frame.astype(np.uint8)



def GetSequenceNames(dirName):
    sequences = ([ f for f in listdir(dirName) 
                 if isfile(join(dirName, f)) and splitext(f)[1] == '.img' ])
    return sequences



def GetRandomSequence():
    sequence = []
    sequence_list = GetSequenceList()
    idx = np.random.randint(0, len(sequence_list)-1)
    sequence.append(sequence_list[idx])
    return sequence


def GetSequence(idx):
    sequence_list = GetSequenceList()
    return filter(lambda x: x['ref'] == idx, sequence_list)




def GetSequenceList():
    sequence_list = [{
	'ref':2,
	'name':'smalllanding4.img', 
    'start_frame':150, 
    'end_frame':450, 
    'x_center':168, 
    'y_center':100, 
    'half_width':12, 
    'half_height':2
},{
	'ref':3,
	'name':'smalllanding4.img', 
	'name':'brwncamp1.img', 
    'start_frame':390, 
    'end_frame':570, 
    'x_center':39, 
    'y_center':148, 
    'half_width':37, 
    'half_height':22
},{
	'ref':4,
	'name':'brwncamp2.img', 
    'start_frame':100, 
    'end_frame':215, 
    'x_center':122, 
    'y_center':139, 
    'half_width':7, 
    'half_height':7
},{
	'ref':5,
	'name':'brwncamp2.img', 
    'start_frame':55, 
    'end_frame':215, 
    'x_center':180, 
    'y_center':136, 
    'half_width':5, 
    'half_height':5
},{
	'ref':6,
	'name':'brwncamp2.img', 
    'start_frame':456, 
    'end_frame':600, 
    'x_center':277, 
    'y_center':137, 
    'half_width':8, 
    'half_height':4
},{
    'ref':7,
	'name':'brwncamp3.img', 
    'start_frame':400, 
    'end_frame':675, 
    'x_center':36, 
    'y_center':143, 
    'half_width':35, 
    'half_height':28
},{
    'ref':8,
	'name':'brwncamp5.img', 
    'start_frame':50, 
    'end_frame':180, 
    'x_center':84, 
    'y_center':125, 
    'half_width':16, 
    'half_height':8
},{
    'ref':9,
	'name':'brwncamp6.img', 
    'start_frame':420,
    'end_frame':800, 
    'x_center':239, 
    'y_center':116, 
    'half_width':20, 
    'half_height':11
},{
	'ref':10,
	'name':'brwncamp8.img', 
    'start_frame':50,
    'end_frame':250, 
    'x_center':72, 
    'y_center':127, 
    'half_width':3, 
    'half_height':10
},{
	'ref':12,
	'name':'SBPATT5.img', 
    'start_frame':810,
    'end_frame':900, 
    'x_center':48, 
    'y_center':128, 
    'half_width':7, 
    'half_height':20
},{
	'ref':13,
	'name':'vons5.img', 
    'start_frame':510,
    'end_frame':700, 
    'x_center':126, 
    'y_center':170, 
    'half_width':54, 
    'half_height':25
}]
    return sequence_list
