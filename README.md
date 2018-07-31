# Feature-Engineering-using-Image-Processing-Techniques
This repository includes Feature Engineering by Python.

**:one: Check video with the frame(FPS)**
-----------------------------------------
```python
# import the libraries for run all sections
import cv2
import numpy as np
import pylab as pl
import pandas as pd
from skimage import color
from scipy import ndimage as ndi
import matplotlib.image as mpimg       
from matplotlib import patches
import matplotlib.pyplot as plt
import glob
import sys
import os

def get_fps(src_dir):
    video = cv2.VideoCapture(src_dir);
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    video.release(); 

src_dir = "/media/rezwan/Study/Thesis/Feature_Extraction_Code/Sample_Videos"

sourceDir = src_dir + '/*.mp4'

vList = glob.glob(sourceDir)
dataFrameArr = []

for i in range(len(vList)):
    vDirName = vList[i]
    head, tail = os.path.split(vDirName)
    print(tail)
    
    # Call `get_fps` function for calculate the frame per second(fps) of a video.
    get_fps(vDirName)

```


**:two: Extract images from video.** [Source](https://gist.github.com/JacopoDaeli/1788da2cef6217549a440ee186d47f68)
-----------------------------------------
```python
def video_to_frames(video_filename, dst_File, tail):
    """Extract frames from video"""
    
    path = dst_File + "/" + tail[:-4]
    print(path)
    
    os.mkdir(path)
    
    cap = cv2.VideoCapture(video_filename)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frames = []
    if cap.isOpened() and video_length > 0:
        frame_ids = [0]
        if video_length >= 4:
            frame_ids = [0, 
                         round(video_length * 0.25), 
                         round(video_length * 0.5),
                         round(video_length * 0.75),
                         video_length - 1]
        count = 0
        success, image = cap.read()
        while success:
            if count in frame_ids:
                frames.append(image)
            success, image = cap.read()

            cv2.imwrite(os.path.join(path, str(count) + '.jpg'), image)

            count += 1
#     return frames

dst_File = "/media/rezwan/Study/Thesis/Feature_Extraction_Code/Images"

src_dir = "/media/rezwan/Study/Thesis/Feature_Extraction_Code/Sample_Videos"

sourceDir = src_dir + '/*.mp4'

vList = glob.glob(sourceDir)

dataFrameArr = []

for i in range(len(vList)):
    vDirName = vList[i]
    head, tail = os.path.split(vDirName)
    # get frames from video
    video_to_frames(vDirName, dst_File, tail)
```

## Name of Inputs images:
  - S85_F56_GL=12.9-850LEDFon_F=100
  - S85_F56_GL=12.9-850LEDFon_F=105
  - S91_F46_GL=7.9-940LEDFon_F=115
  - S91_F46_GL=7.9-940LEDFon_F=120
  - S140_M75_GL=11.5-850LEDFon_F=120
  - S140_M75_GL=11.5-940LEDFon_F=100


**(1) Local Binary Patterns(LBP).** [Source](https://github.com/arsho/local_binary_patterns/blob/master/lbp.py)
```python

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''
     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    
    '''    
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def LBP(img):
    height, width, channel = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_lbp = np.zeros((height, width,3), np.uint8)
    
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

    return img_lbp

if __name__ == '__main__':
    
    path = "/media/rezwan/Study/Thesis/Feature_Extraction_Code/dataset/images"
    dst_path = "/media/rezwan/Study/Thesis/Feature_Extraction_Code/dataset/LBP"
    
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    for ff in os.listdir(path):
        imgg = cv2.imread(os.path.join(path,ff))

        # Read the orginal images
#         cv2.imshow(str(ff),imgg)
#         if cv2.waitKey(0) & 0xff == 27:
#             cv2.destroyAllWindows()


        # Apply operation on Images (LBP)
        res1 = LBP(imgg)
        
        
        cv2.imwrite(os.path.join(dst_path, str(ff)[:-4] + '_lbp.jpg'), res1)

```
 - Inputs:

<p align="center">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43467872-873b029e-9504-11e8-8bf5-f7d92a4d3765.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43467874-879b99ba-9504-11e8-8afa-0c7915adce9b.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43467887-8f8d1bd0-9504-11e8-9ce9-6afe37b80182.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43467888-8fee259c-9504-11e8-9294-9ef35199ad9c.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43481551-0f2ba3e8-9528-11e8-8b0c-1cb365b931cc.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43468021-ca6b0442-9504-11e8-93d9-84001bde3f4e.jpg">
</p>

 - Outputs: 
 
 <p align="center">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43480948-78197de6-9526-11e8-9fcb-7e25e3bedf3c.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43480952-79e41578-9526-11e8-8f7a-cef791f7381c.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43480958-7c4fa4da-9526-11e8-8a48-094cd2fd0aae.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43480965-7f2a8184-9526-11e8-8844-17279ea5239d.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43481650-5ac926a4-9528-11e8-91a2-8e45e9bf10ab.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43480970-83454632-9526-11e8-8ef7-c7001532f28d.jpg">
</p>
