import numpy
import argparse
import os
import cv2
import json
from json import JSONEncoder
import detect


# write ndarray to json
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_parser():
    parser = argparse.ArgumentParser(description="Graffiti Detector")
    parser.add_argument("-i", "--input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image directory")
    return parser


def inference(intput='../graffiti_test_data/test.jpg',
              out_dir=None, 
              model='graffiti_v5s.pt',
              infer_size=[640, 640]
              ):

    # output: ndarray
    # bboxes: list array (n*4), [x1,y1,x2,y2]
    output, bboxes = detect.run(source=intput, weights=model, imgsz=infer_size)

    out_dir = os.path.split(intput)[0] if not out_dir else out_dir
    name, ext = os.path.splitext(os.path.basename(intput))  
    out_name = name + "_pred" + ext
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, output)

    # write into json file. JSON in [key:value] format
    json_path = os.path.join(out_dir, name+".json")
    if len(output.shape) == 2:
        imgsz = {"height": output.shape[0], "width:": output.shape[1]}
    else:
        imgsz = {"height": output.shape[0], "width:": output.shape[1], "nchannel": output.shape[2]}
    lable_id = [0]*len(bboxes)
    class_name = {'0': "graffiti"}

    json_dict = {"imgsz": imgsz}
    json_dict["bbox"]=bboxes
    json_dict["label_id"]=lable_id
    json_dict["class name"]=class_name
    json_dict["img"]=output

    # # dict list 
    # js_imgsz = {"imgsz": imgsz}
    # js_bbox = {"bbox": bboxes}
    # js_lable_id = {"label_id": lable_id}
    # js_class_name = {"class name": class_name}
    # js_img_np = {"img": output}
    # json_dict = []
    # json_dict.append(js_imgsz)
    # json_dict.append(js_bbox)
    # json_dict.append(js_lable_id)
    # json_dict.append(js_class_name)
    # json_dict.append(js_img_np)


    print("Write data into JSON file: ", json_path)
    with open(json_path, "w") as write_file:
        json.dump(json_dict, write_file, indent=4, cls=NumpyArrayEncoder)

    print("Finish writing JSOM file.")


if __name__ == "__main__": 
    args = get_parser().parse_args()
    assert os.path.isfile(args.input)
    if args.output != None:
        assert os.path.isdir(args.output)
    inference(args.input, args.output)

