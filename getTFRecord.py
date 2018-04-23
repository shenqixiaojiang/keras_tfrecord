import numpy as np
import os
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers
from keras.callbacks import Callback

from tensorflow.contrib.learn.python.learn.datasets import mnist

if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend, '
                       'because it requires TFRecords, which '
                       'are not supported on other platforms.')

sess = K.get_session()


cache_dir = os.path.expanduser(
    os.path.join('~', '.keras', 'datasets', 'MNIST-data'))
data = mnist.read_data_sets(cache_dir, validation_size=0)

print data.train.images.shape,data.train.labels.shape

def createTFRecord(filename,data,label):
    print("Converting data into %s ..." % filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index, img in enumerate(data):
        if index == 0:
            print img.shape
            print label[index]
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label[index]])), ##int64 to store the label
            'img_raw': tf.train.Feature(float_list=tf.train.FloatList(value=img)),      ##float32 to store the image
        }))
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()

createTFRecord('./test.tfrecords',data.test.images,data.test.labels)
createTFRecord('./train.tfrecords',data.train.images,data.train.labels)
