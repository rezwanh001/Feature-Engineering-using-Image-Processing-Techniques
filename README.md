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


**(2) Scale-Invariant Feature Transform(SIFT).** [Source](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html)
```python
def SIFT(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
#     kp = sift.detect(gray,None)
    kp = sift.detect(gray)
#     kp, des = sift.detectAndCompute(gray,None)

    img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img


if __name__ == '__main__':
    
    path = "/media/rezwan/Study/Thesis/Feature_Extraction_Code/dataset/images"
    dst_path = "/media/rezwan/Study/Thesis/Feature_Extraction_Code/dataset/SIFT"
    
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    for ff in os.listdir(path):
        imgg = cv2.imread(os.path.join(path,ff))

        # Read the orginal images
#         cv2.imshow(str(ff),imgg)
#         if cv2.waitKey(0) & 0xff == 27:
#             cv2.destroyAllWindows()


        # Apply operation on Images (LBP)
        res1 = SIFT(imgg)
        
        
        cv2.imwrite(os.path.join(dst_path, str(ff)[:-4] + '_sift.jpg'), res1)
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
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502535-33ae4856-957d-11e8-92b9-8cecbb049afc.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502536-33f6a2e0-957d-11e8-86aa-c15b1687755d.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502539-372d65b6-957d-11e8-96c8-2a35f4fba5a9.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502541-37733046-957d-11e8-855d-b1b5916fe9a9.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502546-3c188e84-957d-11e8-874b-358bb3b82274.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502548-3c7717ec-957d-11e8-935f-569f806f07b8.jpg">
</p>

**(3) Gabor Filter.** [Source](https://github.com/Shikhargupta/computer-vision-techniques/blob/master/GaborFilter/gabor.py)
```python
# define gabor filter bank with different orientations and at different scales
def build_filters():
    filters = []
    ksize = 9
    #define the range for theta and nu
    for theta in np.arange(0, np.pi, np.pi / 8):
        for nu in np.arange(0, 6*np.pi/4 , np.pi / 4):
            kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, nu, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

#function to convolve the image with the filters
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

if __name__ == '__main__':

    #instantiating the filters
    filters = build_filters()

    f = np.asarray(filters)

#     #reading the input image
#     imgg = cv2.imread(test,0)
    
    
    path = "/media/rezwan/Study/Thesis/Feature_Extraction_Code/dataset/images"
    dst_path = "/media/rezwan/Study/Thesis/Feature_Extraction_Code/dataset/GaborFilter"
    
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    for ff in os.listdir(path):
        imgg = cv2.imread(os.path.join(path,ff))

        # Read the orginal images
        cv2.imshow(str(ff),imgg)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()


        # Apply operation on Images (Gabor Filter)
        res1 = process(imgg, f)
        
        
        cv2.imwrite(os.path.join(dst_path, str(ff)[:-4] + '_Gabor.jpg'), res1)
        
        cv2.imshow(str(ff)[:-4] + '_Gabor.jpg',res1)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
    

        #initializing the feature vector
        feat = []

        #calculating the local energy for each convolved image
        for j in range(40):
            res = process(imgg, f[j])
            temp = 0
            for p in range(128):
                for q in range(128):
                    temp = temp + res[p][q]*res[p][q]
            feat.append(temp)
        #calculating the mean amplitude for each convolved image	
        for j in range(40):
            res = process(imgg, f[j])
            temp = 0
            for p in range(128):
                for q in range(128):
                    temp = temp + abs(res[p][q])
            feat.append(temp)
        #feat matrix is the feature vector for the image
        print(np.array(feat))
        del feat
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
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502872-a730f52a-957e-11e8-99cb-d68255ad23a2.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502873-a7777d1a-957e-11e8-8a8c-d2a82421b5e5.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502874-a7c006f2-957e-11e8-86c5-9c1c786db387.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502842-876d11b0-957e-11e8-8398-5e54d7b364ae.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502857-94e9e5d4-957e-11e8-9d00-e31164081775.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43502858-9530fdb6-957e-11e8-88ae-454f3ef4f047.jpg">
</p>
  
 **(4) Harris Corner Detection.** [Source](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html) 
 ```python
 def HarrisCorner(img):
    org_img = img

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    
    return img
    
#     cv2.imwrite(dst_file + '/150_HarrisCorner.jpg',img)

#     cv2.imshow('HarrisCorner',img)
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()



if __name__ == '__main__':
    
    path = "/media/rezwan/Study/Thesis/Feature_Extraction_Code/dataset/images"
    dst_path = "/media/rezwan/Study/Thesis/Feature_Extraction_Code/dataset/HarrisCorner"
    
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    for ff in os.listdir(path):
        imgg = cv2.imread(os.path.join(path,ff))

        # Read the orginal images
#         cv2.imshow(str(ff),imgg)
#         if cv2.waitKey(0) & 0xff == 27:
#             cv2.destroyAllWindows()


        # Apply operation on Images (LBP)
        res1 = HarrisCorner(imgg)
        
        
        cv2.imwrite(os.path.join(dst_path, str(ff)[:-4] + '_HCorner.jpg'), res1)
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
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43503079-91a84bda-957f-11e8-9ff7-03823e041a4e.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43503080-91f10c4e-957f-11e8-8d8c-8ec712203861.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43503081-9237514a-957f-11e8-9a4d-003b615adad4.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43503082-927c23ba-957f-11e8-9f9c-5b5788942ecb.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43503083-92c4aa18-957f-11e8-90e8-d785828e494d.jpg">
  <img width="142" height="90" src="https://user-images.githubusercontent.com/15044221/43503084-930f16d4-957f-11e8-8d3a-f2c5eea3d295.jpg">
</p>
 
