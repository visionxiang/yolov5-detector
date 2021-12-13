# Detect anything by YOLOv5

Using yolov5 detector to detect anything




## Use Case

### Case 1: Fish detector by yolov5

1, Download selected classes images (fish) from [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/download.html).

- construct a image file list with specific categories and download them.    
- The [official download script](https://raw.githubusercontent.com/openimages/dataset/master/downloader.py).    
- Some dependencies: `pip install tqdm boto3 botocore requests`;    
- Download by: `python downloader.py $IMAGE_LIST_FILE --download_folder=$DOWNLOAD_FOLDER --num_processes=5`  

Some changes have been made to adjust the dataset to meet yolov5 requirements, please refer to `open-images-downloader`.

Reference: 
[Download Selected classes Images](https://github.com/irvingzhang0512/open-images-downloader)  
[Intro of OpenImages](https://blog.csdn.net/irving512/article/details/116180438)  
[Intro of dataset categories](https://www.pianshen.com/article/7415336050/)
[Other download manner](https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer)


2, Train the detector with yolov5  









Detector based on YOLOv5:





Installation:

Install ffmpeg in docker and build in Ubuntu 18.04, it would stop with select of geographic area on Ubuntu 18.04 or above, you can install it by specifying ENV DEBIAN_FRONTEND=noninteractive as shown below.
```
FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ffmpeg
```

If you cannot intall the ffmpeg, you can docker run the container and install it, then backup the container and save this image (with `docker commit`)


ref: https://stackoverflow.com/questions/44915994/how-can-i-install-ffmpeg-to-my-docker-image






