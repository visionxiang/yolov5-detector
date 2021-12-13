# Detect Anything by YOLOv5

Using yolov5 detector to detect anything




## Use Case

### Case 1: Fish detector by YOLOv5

#### 1) Download selected classes images (fish) from [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/download.html).

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


#### 2) Train the detector with yolov5  







Fish_Counter_YOLOv5L-main:

conda create -n yoloenv python=3.6
cd yolov5
pip install -r requirements.txt



Usage:
detect.py runs inference on a variety of sources, downloading models automatically 
from the latest YOLOv5 release and saving results to runs/detect.

python detect.py --source 0  # webcam
                            img.jpg  # image
                            vid.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream


For example:
python detect.py --source fishimg/fish_3.jpg --weights best.pt --hide-labels --hide-conf --conf-thres=0.5
python detect.py --source fishimg/ --weights best.pt --hide-labels --hide-conf --conf-thres=0.6
python detect.py --source AZUREData.mp4 --weights best.pt --hide-labels --hide-conf --conf-thres=0.6
python detect_count.py --source fishimg/ --weights best.pt --hide-labels --hide-conf --conf-thres=0.6
python detect_count.py --source AZUREData.mp4 --weights best.pt --hide-labels --hide-conf --conf-thres=0.6


Train:
python train.py --img 640 --batch 16 --epochs 300 --data './fishimg/fish_data.yaml' --weights yolov5s.pt --cfg ./models/fish_yolov5s.yaml --cache


Auto track and visualize 
1) tensorboard.  Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
2) pip install wandb


yolov5s_bassfish_det.pt: Fish detection counter yolov5l github


https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.pt 


python detect.py --source fishimg/img/ --weights v5s_best.pt --hide-labels --hide-conf --conf-thres=0.5 --name result



python detect.py --source fishimg/img/ --weights v5s_best.pt --hide-labels --hide-conf --conf-thres=0.5 --name result





---- YOLOv5
---- Download
---- Ownfiles





Installation:

Install ffmpeg in docker and build in Ubuntu 18.04, it would stop with select of geographic area on Ubuntu 18.04 or above, you can install it by specifying ENV DEBIAN_FRONTEND=noninteractive as shown below.
```
FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ffmpeg
```

If you cannot intall the ffmpeg, you can docker run the container and install it, then backup the container and save this image (with `docker commit`)


ref: https://stackoverflow.com/questions/44915994/how-can-i-install-ffmpeg-to-my-docker-image






