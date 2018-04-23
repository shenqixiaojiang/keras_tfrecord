import numpy as np
import os
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers
from keras.callbacks import Callback
from keras.layers import Dense, Dropout,GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from tensorflow.contrib.learn.python.learn.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend, '
                       'because it requires TFRecords, which '
                       'are not supported on other platforms.')


class EvaluateInputTensor(Callback):
    """ Validate a model which does not expect external numpy data during training.

    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
    Instead, create a second model for validation which is also configured
    with input tensors and add it to the `EvaluateInputTensor` callback
    to perform validation.

    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.

    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, model, steps, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix
        print("init")

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        print("set_weights")
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)


def cnn_layers(x_train_input):
    x = layers.Conv2D(32, (3, 3),
                      activation='relu', padding='valid')(x_train_input)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x_train_out = layers.Dense(num_classes,
                               activation='softmax',
                               name='x_train_out')(x)
    
    return x_train_out

sess = K.get_session()

batch_size = 100
image_shape = (28, 28, 1)
batch_shape = (batch_size,) + image_shape
epochs = 20
num_classes = 10
data_augmentation = True

# The capacity variable controls the maximum queue size
# allowed when prefetching data for training.
capacity = 10000

# min_after_dequeue is the minimum number elements in the queue
# after a dequeue, which ensures sufficient mixing of elements.
min_after_dequeue = 3000

# If `enqueue_many` is `False`, `tensors` is assumed to represent a
# single example.  An input tensor with shape `[x, y, z]` will be output
# as a tensor with shape `[batch_size, x, y, z]`.
#
# If `enqueue_many` is `True`, `tensors` is assumed to represent a
# batch of examples, where the first dimension is indexed by example,
# and all members of `tensors` should have the same size in the
# first dimension.  If an input tensor has shape `[*, x, y, z]`, the
# output will have shape `[batch_size, x, y, z]`.
enqueue_many = True

def distorted_inputs(image):
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    distorted_image = tf.image.per_image_standardization(distorted_image)
    return distorted_image

def tfrecord_to_data(filename,data_aug=False):
    # generate a queue with a given file name
    print("reading tfrecords from {}".format(filename))
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([784], tf.float32),
    })
    image = features['img_raw']
    image = tf.reshape(image, shape=image_shape) 
    if data_aug == True:
        image = distorted_inputs(image)
    return image, features['label']

training_filenames = 'data/train.tfrecords'
filenames = training_filenames
train_data, train_label = tfrecord_to_data(filenames,data_aug=data_augmentation)

x_train_batch, y_train_batch = tf.train.shuffle_batch(
    tensors=[train_data, train_label],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue,
    #enqueue_many=enqueue_many, 
    num_threads=4
    )

y_train_batch = tf.one_hot(y_train_batch, num_classes)

x_batch_shape = x_train_batch.get_shape().as_list()
y_batch_shape = y_train_batch.get_shape().as_list()

model_input = layers.Input(tensor=x_train_batch)
model_output = cnn_layers(model_input)
train_model = keras.models.Model(inputs=model_input, outputs=model_output)

# Pass the target tensor `y_train_batch` to `compile`
# via the `target_tensors` keyword argument:

learning_rate = 0.0125
warmup_epochs = 5
weight_decay = 0.00005
opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.9)

train_model.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[y_train_batch])
train_model.summary()

test_data, test_label = tfrecord_to_data(r'data/test.tfrecords')

x_test_batch, y_test_batch = tf.train.batch(
    tensors=[test_data, test_label],
    batch_size=batch_size,
    capacity=capacity,
    #enqueue_many=enqueue_many, 
    num_threads=4)

# Create a separate test model
# to perform validation during training
y_test_batch = tf.one_hot(y_test_batch, num_classes)

x_test_batch_shape = x_test_batch.get_shape().as_list()
y_test_batch_shape = y_test_batch.get_shape().as_list()

test_model_input = layers.Input(tensor=x_test_batch)
test_model_output = cnn_layers(test_model_input)
test_model = keras.models.Model(inputs=test_model_input, outputs=test_model_output)

print x_test_batch.shape,y_test_batch.shape

# Pass the target tensor `y_test_batch` to `compile`
# via the `target_tensors` keyword argument:
test_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'],
                   target_tensors=[y_test_batch])

# Fit the model using data from the TFRecord data tensors.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)

##Here, 60000 is the length of training data.
train_model.fit(epochs=epochs,steps_per_epoch=int(np.ceil(60000 / float(batch_size))),
               verbose=1,callbacks=[EvaluateInputTensor(test_model, steps=100)])

# Save the model weights.
train_model.save_weights('saved_wt.h5')

# Clean up the TF session.
coord.request_stop()
coord.join(threads)
K.clear_session()
