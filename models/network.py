import tensorflow as tf
import numpy as np
import sys, re
sys.path.append('/home/chzze/bitbucket/Mastoid_rls')

from tensorflow.contrib import layers
from data.setup import config

IMAGE_WIDTH = 256
IMAGE_HEIGHT = int(IMAGE_WIDTH * float(config.image_prop))


def design_scope(class_name):
    model_scope = re.sub('Inference', '', class_name)
    classifier_scope = re.sub('Model', 'Classifier', model_scope)
    return model_scope, classifier_scope


def resnet_block(inputs, in_channels, out_channels, use_attention, last_block, is_training):
    conv = layers.conv2d(inputs=inputs, num_outputs=in_channels, kernel_size=1, stride=1, activation_fn=None)
    conv = tf.nn.relu(layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training))

    conv = layers.conv2d(inputs=conv, num_outputs=in_channels, kernel_size=3, stride=1, activation_fn=None)
    conv = tf.nn.relu(layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training))

    conv = layers.conv2d(inputs=conv, num_outputs=out_channels, kernel_size=1, stride=1, activation_fn=None)
    conv = layers.batch_norm(inputs=conv, center=True, scale=True, is_training=is_training)

    def attention_block(inputs, reduction_ratio=4):
        squeeze = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)  # global average pooling
        excitation = layers.fully_connected(inputs=squeeze,
                                            num_outputs=squeeze.shape[-1].value // reduction_ratio,
                                            activation_fn=tf.nn.relu)
        excitation = layers.fully_connected(inputs=excitation, num_outputs=squeeze.shape[-1].value,
                                            activation_fn=tf.nn.sigmoid)

        outputs = inputs * excitation
        return outputs

    if use_attention:
        conv = attention_block(inputs=conv, reduction_ratio=16)

    if not inputs.shape[-1].value == out_channels:
        inputs = layers.conv2d(inputs=inputs, num_outputs=out_channels,
                               kernel_size=1, stride=1, activation_fn=None)
        inputs = layers.batch_norm(inputs=inputs, center=True, scale=True, is_training=is_training)

    if last_block:
        return conv + inputs
    else:
        return tf.nn.relu(conv + inputs)


def calculate_accuracy(prob, label):
    predicted = tf.cast(prob > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, label), dtype=tf.float32))
    return accuracy


def get_global_vars(scope_list):
    _vars = []
    for scope in scope_list:
        _vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    return _vars


def get_prep_conv_vars(scope1, scope2):
    scope_list = [scope1, scope2]
    _vars = get_global_vars(scope_list)
    return _vars


def get_prep_conv_train_vars(scope1, scope2):
    scope_list = [scope1, scope2]
    _vars = []
    for scope in scope_list:
        _vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    return _vars


class InferenceModel51():  # The CNN for single view (AP)
    def __init__(self, trainable=False):
        self.model_scope, self.classifier_scope = design_scope(class_name=type(self).__name__)

        self.img_h, self.img_w, self.img_c = [IMAGE_HEIGHT, IMAGE_WIDTH, 1]
        self.class_num = 1
        self.images = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, self.img_c])
        self.labels = tf.placeholder(tf.int64, shape=[None])  # sparse index
        self.is_training = tf.placeholder(tf.bool, shape=None)

        tf.add_to_collection('images', tf.boolean_mask(self.images, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images, tf.equal(self.labels, 1), name='positive'))

        features_2d = self.images
        with tf.variable_scope(self.model_scope):
            for in_channels, out_channels in zip([64,96,128,160,192], [128,160,192,224,256]):
                features_2d = resnet_block(inputs=features_2d,
                                           in_channels=in_channels, out_channels=out_channels,
                                           use_attention=True, last_block=False,
                                           is_training=self.is_training)
                features_2d = layers.max_pool2d(inputs=features_2d, kernel_size=2, stride=2)

        with tf.variable_scope(self.classifier_scope):
            for in_channels, out_channels in zip([224], [self.class_num]):
                self.features_2d = resnet_block(inputs=features_2d,
                                                in_channels=in_channels, out_channels=out_channels,
                                                use_attention=True, last_block=True,
                                                is_training=self.is_training)
            self.logits = tf.reduce_logsumexp(self.features_2d, axis=[1,2], keepdims=False)

        self.prob = tf.nn.sigmoid(self.logits)

        labels = tf.expand_dims(self.labels, axis=-1)
        self.accuracy = calculate_accuracy(self.prob[:, 0], tf.cast(self.labels, dtype=tf.float32))
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=self.logits)
        self.loss = tf.reduce_mean(bce)

        self.local_class = tf.image.resize_bilinear(images=self.features_2d, size=[self.img_h, self.img_w])
        self.local = tf.nn.relu(self.local_class)

        if trainable:
            self.global_step, self.global_epoch, self.train = \
                training_option(self.loss, learning_rate=0.01, decay_steps=5000, decay_rate=0.94,
                                decay=0.9, epsilon=0.1)


class InferenceModel52():  # The CNN for a single view (Lateral)
    def __init__(self, trainable=False):
        self.model_scope, self.classifier_scope = design_scope(class_name = type(self).__name__)

        self.img_h, self.img_w, self.img_c = [IMAGE_WIDTH, IMAGE_WIDTH, 1]
        self.class_num = 1
        self.images = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, self.img_c])
        self.labels = tf.placeholder(tf.int64, shape=[None])  # sparse index
        self.is_training = tf.placeholder(tf.bool, shape=None)

        tf.add_to_collection('images', tf.boolean_mask(self.images, tf.equal(self.labels, 0), name='negative'))
        tf.add_to_collection('images', tf.boolean_mask(self.images, tf.equal(self.labels, 1), name='positive'))

        features_2d = self.images
        with tf.variable_scope(self.model_scope):
            for in_channels, out_channels in zip([64,96,128,160,192], [128,160,192,224,256]):
                features_2d = resnet_block(inputs=features_2d,
                                           in_channels=in_channels, out_channels=out_channels,
                                           use_attention=True, last_block=False,
                                           is_training=self.is_training)
                features_2d = layers.max_pool2d(inputs=features_2d, kernel_size=2, stride=2)

        with tf.variable_scope(self.classifier_scope):
            for in_channels, out_channels in zip([224], [self.class_num]):
                self.features_2d = resnet_block(inputs=features_2d,
                                                in_channels=in_channels, out_channels=out_channels,
                                                use_attention=True, last_block=True,
                                                is_training=self.is_training)
            self.logits = tf.reduce_logsumexp(self.features_2d, axis=[1,2], keepdims=False)

        self.prob = tf.nn.sigmoid(self.logits)

        labels = tf.expand_dims(self.labels, axis=-1)
        self.accuracy = calculate_accuracy(self.prob[:, 0], tf.cast(self.labels, dtype=tf.float32))
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=self.logits)
        self.loss = tf.reduce_mean(bce)

        self.local_class = tf.image.resize_bilinear(images=self.features_2d, size=[self.img_h, self.img_w])
        self.local = tf.nn.relu(self.local_class)

        if trainable:
            self.global_step, self.global_epoch, self.train = \
                training_option(self.loss, learning_rate=0.01, decay_steps=5000, decay_rate=0.94,
                                decay=0.9, epsilon=0.1)


class InferenceModel60:  # The CNN for multiple views
    def __init__(self, trainable=False):
        self.model_scope, self.classifier_scope = design_scope(class_name=type(self).__name__)
        self.concat_scope = re.sub('Model', 'Concat', self.model_scope)
        self.img_h, self.img_w, self.img_c = [IMAGE_HEIGHT, IMAGE_WIDTH, 1]
        self.class_num = 1

        self.images0 = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, self.img_c])
        self.images1 = tf.placeholder(tf.float32, shape=[None, self.img_w, self.img_w, self.img_c])

        self.list_images = [self.images0, self.images1]
        self.labels = tf.placeholder(tf.int64, shape=[None])
        self.is_training = tf.placeholder(tf.bool, shape=None)

        self.list_features = []
        pre_model = {self.list_images[0]: 'Model51', self.list_images[1]: 'Model52'}

        for view in self.list_images:
            with tf.variable_scope(pre_model[view]):
                features_2d = view
                for in_channels, out_channels in zip([64, 96, 128, 160, 192], [128, 160, 192, 224, 256]):
                    features_2d = resnet_block(inputs=features_2d,
                                               in_channels=in_channels, out_channels=out_channels,
                                               use_attention=True, last_block=False,
                                               is_training=self.is_training)
                    features_2d = layers.max_pool2d(inputs=features_2d, kernel_size=2, stride=2)
                self.list_features.append(features_2d)

        self.list_locals, self.list_logits = [], []
        pre_classifier = {self.list_features[0]: 'Classifier51', self.list_features[1]: 'Classifier52'}
        for view in self.list_features:
            with tf.variable_scope(pre_classifier[view]):
                for in_channels, out_channels in zip([224], [self.class_num]):
                    classify_block = resnet_block(inputs=view,
                                                  in_channels=in_channels, out_channels=out_channels,
                                                  use_attention=True, last_block=True,
                                                  is_training=self.is_training)
                    log_sum_exp = tf.reduce_logsumexp(classify_block, axis=[1,2], keepdims=False)

                    self.list_locals.append(classify_block)
                    self.list_logits.append(log_sum_exp)

        self.local0_class = tf.image.resize_bilinear(images=self.list_locals[0], size=[self.img_h, self.img_w])
        self.local0 = tf.nn.relu(self.local0_class)

        self.local1_class = tf.image.resize_bilinear(images=self.list_locals[1], size=[self.img_w, self.img_w])
        self.local1 = tf.nn.relu(self.local1_class)

        with tf.variable_scope(self.concat_scope):
            view_concat = tf.concat(self.list_logits, axis=1)
            self.logits = tf.expand_dims(tf.reduce_mean(view_concat, axis=1), axis=-1)
            self.prob = tf.nn.sigmoid(self.logits)

        labels = tf.expand_dims(self.labels, axis=-1)
        self.accuracy = calculate_accuracy(self.prob, tf.cast(self.labels, dtype=tf.float32))
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=self.logits)
        self.loss = tf.reduce_mean(bce)

        if trainable:
            self.global_step, self.global_epoch, self.train = \
                training_option(self.loss, learning_rate=0.005, decay_steps=1000, decay_rate=0.04,
                                decay=0.9, epsilon=0.1)


def training_option(loss, learning_rate=0.01, decay_steps=5000, decay_rate=0.04, decay=0.9, epsilon=0.1):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_epoch = tf.Variable(0, trainable=False, name='global_epoch')

    with tf.variable_scope('reg_loss'):
        reg_loss = 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
    lr_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=tf.train.get_global_step(),
                                         decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate, decay=decay, epsilon=epsilon)
        train = optimizer.minimize(loss+reg_loss, global_step=tf.train.get_global_step())
    return global_step, global_epoch, train


if __name__ == '__main__':
    import argparse, network
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='', epilog=license, add_help=False)
    network_config = parser.add_argument_group('network setting (must be provided)')
    network_config.add_argument('--infer', type=str, dest='infer', default='Model51')
    config, unparsed = parser.parse_known_args()

    cnn = getattr(network, 'Inference'+config.infer)(trainable=True)




