import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    concatenate,
    Input,
    Conv2D,
    MaxPool2D,
    Lambda,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy,
)
from .utils import broadcast_IOU

# --------------------------------------------------------------------------
yolo_max_boxes = 100
yolo_iou_threshold = 0.5
yolo_score_threshold = 0.5
# --------------------------------------------------------------------------
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169), (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


# --------------------------------------------------------------------------
def DarknetConv(x, filters, kernel_size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
    x = Conv2D(filters, kernel_size, strides, padding, use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
    return x


# --------------------------------------------------------------------------
def DarknetRes(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


# --------------------------------------------------------------------------
def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, 2)
    for _ in range(blocks):
        x = DarknetRes(x, filters)
    return x


# --------------------------------------------------------------------------
def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3, )
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)
    x = x_36 = DarknetBlock(x, 256, 8)
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return Model(inputs, (x_36, x_61, x), name=name)


# --------------------------------------------------------------------------
def DarkntTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 512, 3)
    return Model(inputs, (x, x_8), name=name)


# --------------------------------------------------------------------------
def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=x)(x_in)

    return yolo_conv


# --------------------------------------------------------------------------
def YoloConvTiny(filters, name=None):
    def yolo_conv_tiny(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv_tiny


# --------------------------------------------------------------------------
def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 1)
        x = DarknetConv(x, anchors * (5 + classes), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[1],
                                            anchors, classes + 5)))(x)
        return Model(inputs, x, name=name)(x_in)

    return yolo_output


# --------------------------------------------------------------------------
def Yolo_Boxes(pred, anchors, classes):
    grid_size = tf.shpae(pred)[1:3]
    bbox_xy, bbox_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)
    bbox_xy = tf.sigmoid(bbox_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((bbox_xy, bbox_wh), axis=-1)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    bbox_xy = (bbox_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    bbox_wh = tf.exp(bbox_wh) * anchors
    bbox_x1y1 = bbox_xy - bbox_wh / 2
    bbox_x2y2 = bbox_xy + bbox_wh / 2
    bbox = tf.concat([bbox_x1y1, bbox_x2y2], axis=-1)
    return bbox, objectness, class_probs, pred_box


# --------------------------------------------------------------------------
def Yolo_NMS(outputs, anchors, masks, classes):
    boxes, confs, types = [], [], []
    for output in outputs:
        boxes.append(tf.reshpae(output, (tf.shape(output[0])[0], -1, tf.shape(output[0])[-1])))
        confs.append(tf.reshpae(output, (tf.shape(output[1])[0], -1, tf.shape(output[1])[-1])))
        types.append(tf.reshpae(output, (tf.shape(output[2])[0], -1, tf.shape(output[2])[-1])))
    bbox = tf.concat(boxes, axis=1)
    confidence = tf.concat(confs, axis=1)
    class_probs = tf.concat(types, axis=1)
    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shpae(scores)[-1])),
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
    )
    return boxes, scores, classes, valid_detections


# --------------------------------------------------------------------------
def YoloV3():
    return 0
