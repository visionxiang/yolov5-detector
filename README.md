# Detect Anything by YOLOv5

Using yolov5 detector to detect anything. Some cases to show how to use YOLOv5 to build, train and test a model with your own data.

Requirements:

- Ubuntu 18.04
- Nvidia Titan XP
- cuda==11.2





## Case 1: Fish detector by YOLOv5

### 1) Download selected classes images (fish) from [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/download.html).

- construct a image file list with specific categories and download them.    
- The [official download script](https://raw.githubusercontent.com/openimages/dataset/master/downloader.py).    
- Some dependencies: `pip install tqdm boto3 botocore requests opencv-python pandas`;    
- Download by: `python downloader.py $IMAGE_LIST_FILE --download_folder=$DOWNLOAD_FOLDER --num_processes=5`  

Some changes have been made to adjust the dataset to meet yolov5 requirements, please refer to `open-images-downloader`.

Noted:  
- Replace with your selected classes when `python generate_filelist.py {class1} {class2} --filelist filelist.txt --splits train validation test
`, e.g., `Fish Starfish Goldfish Jellyfish`;  
- Change the class name with your selected classes in `convert-to-yolo.py`, e.g., `target_class_names = ["Fish", "Starfish", "Goldfish", "Jellyfish"]`.

Reference:  
-- The download code is from [[download selected classes images]](https://github.com/irvingzhang0512/open-images-downloader)    
-- Introduction of OpenImages dataset: [[1]](https://blog.csdn.net/irving512/article/details/116180438), [[2]](https://www.pianshen.com/article/7415336050/)  
-- Other download manner, e.g. [awscli](https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer)



### 2) Prepare/organize your own data

Organize your own dataset: 2 ways.

**First**, as above 1), write the path of training data into txt files.

```
├── data
│   ├── images # save all images including train, val and test
│   ├── labels # save all annotation files including train, val and test, one image <--> one txt with the same name
│   ├── train.txt # save the path of training images, one image path per line, e.g., /home/data/images/test01.jpg
│   ├── validation.txt # save the path of validation images
│   ├── test.txt # save the path of test images
```

Build your data yaml, e.g., fish_data.yaml. Refer data/coco.yaml, that is:

```
train: /home/data/train.txt   # train images
val: /home/data/validation.txt  # train images 
test: /home/data/test.txt

# Classes
nc: 4  # number of classes for your own dataset
names: ['Fish', 'Starfish', 'Goldfish', 'Jellyfish'] 
```

The code will find the labels automatically, by replacing the `images` with `labels` in the image path.  
Noted that, there is a space after 'train:'.

**Second**, directly input the folders of the training files.
```
root
├── data
│   ├── images
│   │   ├── train
│   │   ├── val
│   │   └── test
│   └── labels
│       ├── train
│       ├── val
│       └── test
└── yolov5
```
```
train: /home/data/train   # train images
val: /home/data/val  # train images 
test: /home/data/test

# Classes
nc: 4  # number of classes for your own dataset
names: ['Fish', 'Starfish', 'Goldfish', 'Jellyfish'] 
```

Noted that, the bounding box for yolo is: `class_id, cx, cy, w, h` in [0,1].



### 3) Train the detector with yolov5  


1, Download [YOLOv5](https://github.com/ultralytics/yolov5) files and config the environment.   
```
cd yolov5
conda create -n yolov5 python=3.6
conda activate yolov5
pip install -r requirements.txt
```

2, Test the yolov5
```
# detect.py runs inference on a variety of sources, downloading models automatically 
# from the latest YOLOv5 release and saving results to runs/detect.

python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream


# For example:
python detect.py --source testimg/test_0.jpg --weights yolov5s.pt --hide-labels --hide-conf --conf-thres=0.5
python detect.py --source testimg/ --weights yolov5m.pt --hide-labels --hide-conf --conf-thres=0.6
python detect.py --source test.mp4 --weights yolov5l.pt --hide-labels --hide-conf --conf-thres=0.6
```

3, Train yolov5 with your own data

First, modify the model files, e.g. `yolov5s.yaml` as below:  
```
# Parameters
nc: 4  # number of classes
```
Name it as fish_yolov5s.yaml. Then train the detector:  

```
python train.py --img 640 --batch-size 16 --epochs 300 --data './data/fish_data.yaml' --weights yolov5s.pt --cfg ./models/fish_yolov5s.yaml --cache
```

During the training, you can visulize the training status/process using Tensorboard or wandb. 

```
1) tensorboard. Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
2) pip install wandb
```

If errors for wandb, you need to initilize the wandb: `wandb init`. 


4, Test the newly trained detector  

After training, the trained weights (best.pt and last.pt) will be placed in `runs/train/exp/weights`. 
```
python detect.py --source fishimg/img/ --weights best.pt --hide-labels --hide-conf --conf-thres=0.5 --name result
```



Reference: [[1]](http://article.docway.net/it/details/6072e8bc0a6c640f8b462446), [[2]](https://www.mdnice.com/writing/328959e2439045849a06933c6380e148), [Fish detection counter of the Black Sea Bass](https://github.com/OnurcanKoken/Fish_Counter_YOLOv5L)




## Case 2: Graffiti/Tagging detection by YOLOv5

The STORM graffiti/tagging detection dataset can be download [here](https://zenodo.org/record/3238357#.YbXveH1ByYA).   

Then transfer the bounding box annotation in XYXY_ABS mode to yolo format, and then training the detector.





## Others

1, Install `ffmpeg` in docker for the video stream. 

Install ffmpeg in docker and build in Ubuntu 18.04, it would stop with select of geographic area on Ubuntu 18.04 or above, you can install it by specifying ENV DEBIAN_FRONTEND=noninteractive as shown below.

```
FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ffmpeg
```

If you cannot intall the ffmpeg, you can docker run the container and install it, then backup the container and save this image (with `docker commit`)

ref: https://stackoverflow.com/questions/44915994/how-can-i-install-ffmpeg-to-my-docker-image


2, Bounding box annotation 

- [CVAT](https://github.com/openvinotoolkit/cvat): Powerful and efficient Computer Vision Annotation Tool (CVAT), getting start see [here](https://blog.roboflow.com/cvat/).  
- [Roboflow](https://docs.roboflow.com/) 
- [LabelImg](https://github.com/tzutalin/labelImg)



