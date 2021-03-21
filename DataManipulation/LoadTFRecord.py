import tensorflow as tf
import numpy as np
# ----------------------------------------------------------
yolo_max_boxes = 100
FEATURE_MAP = {
    #'image/height' : tf.io.FixedLenFeature([],tf.int64),
    #'image/width' : tf.io.FixedLenFeature([],tf.int64),
    #'image/filename' : tf.io.FixedLenFeature([],tf.string),
    #'image/source_id': tf.io.FixedLenFeature([],tf.string),
    #'image/sha256' : tf.io.FixedLenDigest([],tf.string),
    'image/encoded' : tf.io.FixedLenFeature([],tf.string),
    #'image/format' : tf.io.FixedLenFeature([],tf.string),
    'image/object/bbox/xmin' : tf.io.VarLenFeature([],tf.float32),
    'image/object/bbox/xmax' : tf.io.VarLenFeature([],tf.float32),
    'image/object/bbox/ymin' : tf.io.VarLenFeature([],tf.float32),
    'image/object/bbox/ymax' : tf.io.VarLenFeature([],tf.float32),
    'image/object/class/text':tf.io.VarLenFeature([],tf.string),
    #'image/object/class/label' : tf.io.VarLenFeature([],tf.int64),
    #'image/object/difficult' : tf.io.VarLenFeature([],tf.int64),
    #'image/object/truncated' : tf.io.VarLenFeature([],tf.int64),
    #'image/object/views' : tf.io.VarLenFeature([],tf.string)
}
# ----------------------------------------------------------
def parse_tfrecord(tfrecord,class_tabel,size):
    x = tf.io.parse_single_example(tfrecord,FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'],channels=3)
    x_train = tf.image.resize(x_train,(size,size))
    class_text = tf.sparse.to_dense(
        x['image/object/class/text'],default_value=''
    )
    labels = tf.cast(class_tabel.lookup(class_text),tf.float32)
    y_train = tf.stack([
        tf.sparse.to_dense(x['image/object/bbox/xmin']),
        tf.sparse.to_dense(x['image/object/bbox/ymin']),
        tf.sparse.to_dense(x['image/object/bbox/xmax']),
        tf.sparse.to_dense(x['image/object/bbox/ymax']),
    labels],axis=1)
    padding = [[0,yolo_max_boxes - tf.shape(y_train)[0]],[0,0]]
    y_train = tf.pad(y_train,padding)
    return x_train , y_train

# ----------------------------------------------------------
def load_tfrecord_dataset(filepattern,class_file,size=416):
    LINE_NUMBER = tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file,tf.string,0,tf.int64,LINE_NUMBER,delimiter='\n'),-1)
    files = tf.data.Dataset.list_files(filepattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x : parse_tfrecord(x,class_table,size))
