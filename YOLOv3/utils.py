import tensorflow as tf
import cv2 as cv
import numpy as np
# -------------------------------------------
YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]
# -------------------------------------------
def load_darknet_weights(model,weights_file,tiny=False):
    with open(weights_file,'rb') as wf:
        major , minor , seen , _ = np.fromfile(wf,dtype=np.int32,count=5)
        if tiny:
            layers = YOLOV3_TINY_LAYER_LIST
        else : 
            layers = YOLOV3_LAYER_LIST
        for layer_name in layers:
            sub_model = model.get_layer(layer_name)
            for i , layer in enumerate(layer_name):
                if not layer.name.startswith('conv2d'):
                    continue
                batch_norm = None
                if i + 1 < len(sub_model.layers) and sub_model.layers[i+1].name.startswith('batch_norm') :
                    batch_norm = sub_model.layers[i + 1]
                
                filters = layer.filters
                size = layer.kernel_size[0]
                input_dim = layer.get_input_shape_at(0)[-1]

                if batch_norm in None:
                    conv_bias = np.fromfile(wf,dtype=np.float32,count=filters)
                else :
                    # darknet [beta,gamma,mean,variance]
                    bn_weights = np.fromfile(wf,dtype=np.float32,count=4 * filters)
                    # tf [gamma,beta,mean,variance]
                    bn_weights = bn_weights.reshape((4,filters))[[1,0,2,3]]
                
                
                # darknet shape (output_dim,input_dim,height,width)
                conv_shape = (filters,input_dim,size,size)
                conv_weights = np.fromfile(wf,dtype=np.float32,count=np.product(conv_shape))
                # tf shape (height,width,input_dim,output_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2,3,1,0])

                if batch_norm is None:
                    layer.set_weights([conv_weights,conv_bias])
                else : 
                    layer.set_weights([conv_weights])
                    batch_norm.set_weights(bn_weights)
# -------------------------------------------
def broadcast_IOU(box1,box2):
    # box1 (...,(x1,y1,x2,y2))
    # box2 = (N,(x1,y1,x2,y2))
    box1 = tf.expand_dims(box1,-2)
    box2 = tf.expand_dims(box2,0)
    # new shape (...,N,(x1,y1,x2,y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box1),tf.shape(box2))
    box1 = tf.broadcast_to(box1,new_shape)
    box2 = tf.broadcast_to(box2,new_shape)

    int_width = tf.maximum(tf.minimum(box1[...,2],box2[...,2]) - 
                           tf.maximum(box1[...,0],box2[...,0]) , 0)
    int_height = tf.maximum(tf.minimum(box1[...,3],box2[...,3]) - 
                            tf.maximum(box1[...,1],box2[...,1]), 0)
    
    intersection_area = int_height * int_width
    box1_area = (box1[...,2] - box1[...,0]) * (box1[...,3] - box1[...,1])
    box2_area = (box2[...,2] - box2[...,0]) * (box2[...,3] - box2[...,1])
    return intersection_area / (box1_area + box2_area - intersection_area)
# -------------------------------------------
def draw_outputs(image,outputs,class_names):
    boxes , objectness , classes , nums = outputs
    boxes , objectness , classes , nums = boxes[0] , objectness[0] , classes[0] , nums[0]
    wh = np.flip(image.shape[0:2])
    for i in nums:
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv.rectangle(image,x1y1,x2y2,(255,0,0),2)
        img = cv.putText(img,"{} {:.4f}".format(
            class_names[int(classes[i])],objectness[i]),
            x1y1,cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
    return img
# -------------------------------------------
def draw_labels(x,y,class_names):
    img = x.numpy()
    boxes,classes = tf.split(y,(4,1),axis=-1)
    classes = classes[...,0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv.rectangle(img,x1y1,x2y2,(255,0,0),2)
        img = cv.putText(img,class_names[classes[i]],x1y1,
                        cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
    return img
# -------------------------------------------
def freeze_all(model,frozen=True):
    model.trainable = not frozen
    if isinstance(model,tf.keras.Model):
        for l in model.layers:
            freeze_all(l,frozen)
    
