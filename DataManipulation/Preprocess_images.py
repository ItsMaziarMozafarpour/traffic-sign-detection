import pandas as pd
import os
import cv2 as cv
import tensorflow as tf
from tensorflow.train import Example , Int64List , BytesList , Features , FloatList , Feature 
import xml.etree.cElementTree as ET
from PIL import Image
from tqdm import tqdm
import lxml.etree
import time
from absl import logging , app , flags
from contextlib import ExitStack
import hashlib
# ---------------------------------------------------------------
IMAGES_DIR = '../data/TrainIJCNN2013/'
DATA_INFO_DIR = '../data/TrainIJCNN2013/data_info.csv'
ANNOTATIONS_DIR_PREFIX = "../data/Images"
DESTINATION_DIR = "../data/XML"
CLASS_MAPPING = {
    '0' : 'prohibitory',
    '1' : 'danger' ,
    '2' : 'mandatory' ,
    '3' : 'other' ,
}
TRAIN_TFRECORD_OUTPUT = '../tfrecord/train/'
VALID_TFRECORD_OUTPUT = '../tfrecord/valid/'
DATA_DIR = '../data/'
CLASSES_DIR = '../data/obj.names'
# ---------------------------------------------------------------
# Load csv contains data info(path,bounding box,id,object name)
data_info_df = pd.read_csv(DATA_INFO_DIR)
final_df = pd.DataFrame(columns=data_info_df.columns)
# ---------------------------------------------------------------
# Remove .ppm from images path in dataframe
for i in range(len(data_info_df)):
    data_info_df['path'][i] = data_info_df['path'][i][:-4]
# ---------------------------------------------------------------
# Create jpg images from ppm images
for current_dir , dirs ,files in os.walk(IMAGES_DIR):
    for f in files:
        if f.endswith('.ppm'):
            image_name = f[:-4]
            image = cv.imread(IMAGES_DIR + f)
            single_data_line = data_info_df.loc[data_info_df['path'] == f[:-4]].copy()
            if single_data_line.isnull().values.all():
                os.remove(IMAGES_DIR + f)
            else:
                final_df = final_df.append(single_data_line)
                save_path = "../data/Images/" + image_name + '.jpg'
                if not os.path.isfile(save_path):
                    cv.imwrite(save_path,image)
# ---------------------------------------------------------------
final_df = final_df[~final_df.index.duplicated(keep='first')]
final_df.sort_index(inplace=True)
for i in range(len(final_df)):
    final_df['path'][i] = final_df['path'][i] + '.jpg'
final_df.to_csv('../data/TrainIJCNN2013/final_df.csv',index=False)
# ---------------------------------------------------------------
# Creating annotaion files for each image :
def create_root(filename, width, height):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "Images"
    ET.SubElement(root, "filename").text = (filename)
    ET.SubElement(root, "path").text = "../data/Images/{}".format(filename)
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"
    return root
# ---------------------------------------------------------------
def create_object_annotation(root, voc):
    for ind,voc_label in voc.iterrows():
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text=str(CLASS_MAPPING.get(str(voc_label["id"])))
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label["left"])
        ET.SubElement(bbox, "ymin").text = str(voc_label["top"])
        ET.SubElement(bbox, "xmax").text = str(voc_label["right"])
        ET.SubElement(bbox, "ymax").text = str(voc_label["bottom"])
    if(len(voc)==0):
        print(voc)
        print('no')
    return root
# ---------------------------------------------------------------
def create_file(filename, width, height, voc):
    root = create_root(filename, width, height)
    root = create_object_annotation(root, voc)
    tree = ET.ElementTree(root)
    tree.write("{}/{}.xml".format(DESTINATION_DIR, filename[:-4]))
# ---------------------------------------------------------------
def read_file(filename):
    img = cv.imread("{}/{}".format("../data/Images/", filename))
    w,h=img.shape[:2]
    voc = final_df.loc[final_df.path == filename[:-4]].copy()
    voc["name"] = CLASS_MAPPING.get(str((voc["id"])))
    create_file(filename, w, h, voc)
# ---------------------------------------------------------------
def start():
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
    for filename in os.listdir(ANNOTATIONS_DIR_PREFIX):
          if filename.endswith(".jpg"):
            read_file(filename)


# if __name__ == "__main__":
#     start()
# ---------------------------------------------------------------
train_paths = data_info_df[:700]['path']
valid_paths = data_info_df[700:]['path']

with open('../data/train.txt','w') as train:
    for path in train_paths:
        train.write(path)
        train.write('\n')

with open('../data/valid.txt','w') as val:
    for path in valid_paths:
        val.write(path)
        val.write('\n')
# ---------------------------------------------------------------
# Create TFRecord files :
def _bytes_feature(value : list):
    return Feature(bytes_list=BytesList(value=value))
def _int64_feature(value : list):
    return Feature(int64_list=Int64List(value=value))
def _float_feature(value : list):
    return Feature(float_list=FloatList(value=value))
# ---------------------------------------------------------------
def create_example(annotations,class_map):
    image_path = os.path.join(DATA_DIR,'Images',annotations['filename'])
    image_raw = open(image_path,'rb').read()
    key = hashlib.sha256(image_raw).hexdigest()
    width , height = int(annotations['size']['width']) , int(annotations['size']['height'])
    xmin , xmax , ymin , ymax , classes , classes_text , truncated , views , difficult_obj =\
    [] , [] , [] , [] , [] , [] , [] , [] ,[]
    if 'object' in annotations:
        for obj in annotations['object']:
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))
    example = Example(features=Features(feature={
        'image/height' : _int64_feature(value=[height]),
        'image/width' : _int64_feature(value=[width]),
        'image/filename' : _bytes_feature(value=[annotations['filename'].encode('utf8')]),
        'image/source_id' : _bytes_feature(value=[annotations['filename'].encode('utf8')]),
        'image/sha256' : _bytes_feature(value=[key.encode('utf8')]),
        'image/encoded' : _bytes_feature(value=[image_raw]),
        'image/format' : _bytes_feature(value=['jpg'.encode('utf8')]),
        'image/object/bbox/xmin' : _float_feature(value=xmin),
        'image/object/bbox/xmax' : _float_feature(value=xmax),
        'image/object/bbox/ymin' : _float_feature(value=ymin),
        'image/object/bbox/ymax' : _float_feature(value=ymax),
        'image/object/class/text' : _bytes_feature(value=classes_text),
        'image/object/class/label' : _int64_feature(value=classes),
        'image/object/difficult' : _int64_feature(value=difficult_obj),
        'image/object/truncated' : _int64_feature(value=truncated),
        'image/object/views' : _bytes_feature(value=views)
    }))
    return example
# ---------------------------------------------------------------
def parse_xml(xml):
    if not len(xml):
        return {xml.tag : xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else :
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag : result}
# ---------------------------------------------------------------
def create_tfrecord(data_dir,tfrecord_path,n_shards=10,split=None,class_map_dir=CLASSES_DIR):
    paths = ["{}.tfrecord-{}-of-{}".format(split,idx + 1,n_shards) for idx in range(n_shards)]
    class_map = {name : idx for 
                 idx , name in enumerate(open(class_map_dir).read().splitlines())}
    logging.info("Class mapping loaded : %s",class_map)
    image_list = open(os.path.join(data_dir,"%s.txt" % split)).read().splitlines()
    logging.info("%d images loaded",len(image_list))
    with ExitStack() as stack:
        writers = [stack.enter_context(tf.io.TFRecordWriter(os.path.join(tfrecord_path,path))) 
                   for path in paths]
        for index,img_name in enumerate(image_list):
            shard = index % n_shards
            annotaion_xml = os.path.join(data_dir,"XML",img_name + ".xml")
            annotaion_xml = lxml.etree.fromstring(open(annotaion_xml).read())
            annotation = parse_xml(annotaion_xml)['annotation']
            tfrecord_example = create_example(annotation,class_map)
            writers[shard].write(tfrecord_example.SerializeToString())
    return paths
# ---------------------------------------------------------------