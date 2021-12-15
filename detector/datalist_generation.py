
import numpy as np
import pandas as pd
import os


# BoxMode.XYXY_ABS -> YOLOBBOX
def convert(csv_file, labels_dir):

    # header=0: the first line of csv file is the header name of dataframe default
    csv_reader = pd.read_csv(csv_file, dtype={'filename': str, 'width':int, 'height':int, 'class':str, 'xmin':int, 'ymin':int, 'xmax':int, 'ymax':int})

    ncount = len(csv_reader)
    print(csv_reader.shape)

    # for index, row in csv_reader.iterrows(): print(row['filename'], row['width'])
    for i in range(ncount):
        filename = csv_reader["filename"][i]
        width = csv_reader["width"][i]
        height = csv_reader["height"][i]
        category = csv_reader["class"][i]
        xmin = csv_reader["xmin"][i]
        ymin = csv_reader["ymin"][i]
        xmax = csv_reader["xmax"][i]
        ymax = csv_reader["ymax"][i]

        # BoxMode.XYXY_ABS -> YOLOBBOX  
        dw = 1./width
        dh = 1./height
        cx = (xmin + xmax)/2.0 - 1
        cy = (ymin + ymax)/2.0 - 1
        w = xmax - xmin
        h = ymax - ymin
        cx = cx*dw
        w = w*dw
        cy = cy*dh
        h = h*dh
        if category == "Graffiti":
            label_id = 0
        else:
            label_id = 1

        img_name = filename.split(".")[0]
        with open(os.path.join(labels_dir, f'{img_name}.txt'), 'w') as label_writer:
            label_writer.write(f'{label_id} {cx} {cy} {w} {h}\n')

    print(f'Finished: convert {ncount} labels -> yolo fomat.')


# YOLOBBOX -> BoxMode.XYXY_ABS
def yolobbox2bbox(box, imgh, imgw):
    x, y, w, h = box
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    
    l = int((x - w / 2) * imgw)
    r = int((x + w / 2) * imgw)
    t = int((y - h / 2) * imgh)
    b = int((y + h / 2) * imgh)

    l = max(l, 0)
    r = min(r, imgw - 1)
    t = max(t, 0)
    b = min(b, imgh - 1)

    return [l, t, r, b]


if __name__ == "__main__":

    train_csv_path = "/home/graffiti/Bounding_boxes/train_labels.csv"
    test_csv_path = "/home/graffiti/Bounding_boxes/test_labels.csv"

    labels_dir = 'home/graffiti/labels/test'

    convert(csv_file=train_csv_path, labels_dir=labels_dir)