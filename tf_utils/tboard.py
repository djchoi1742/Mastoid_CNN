import tensorflow as tf
import datetime


class Tensorboard():
    def __init__(self, log_dir='checkpoint', overwrite=True, **kwargs):
        import glob, os

        self.summary_writer_train = tf.summary.FileWriter(os.path.join(log_dir, 'train_logs'))
        self.summary_writer_eval = tf.summary.FileWriter(os.path.join(log_dir, 'test_logs'))
        self.summary_variables = []

        self.performance_per_epoch = []

        self.log_string = ''
        self.start_time = datetime.datetime.now()

        log_files = glob.glob(os.path.join(log_dir, 'train_logs', 'events*')) + \
                    glob.glob(os.path.join(log_dir, 'test_logs', 'events*'))
        if overwrite:
            for file in log_files: os.remove(file)

    def init_scalar(self, collections=None):
        for collection in collections:
            summary_variables = tf.get_collection(collection)
            with tf.variable_scope(collection) as scope:  # using collection name as scope
                for var in summary_variables:
                    tf.summary.scalar(var.name, var)
                    self.summary_variables.append(var)

    def init_images(self, collections=None, num_outputs=1):
        for collection in collections:
            summary_variables = tf.get_collection(collection)
            with tf.variable_scope(collection) as scope:  # using collection name as scope
                for var in summary_variables:
                    tf.summary.image(var.name, var, max_outputs=num_outputs)
                    self.summary_variables.append(var)

    def add_summary(self, sess, feed_dict, log_type='train'):
        images_dict = dict()

        for var, value in feed_dict.items():
            # ex) tf.Variable
            if 'ref' in str(var.dtype) and var in self.summary_variables:
                sess.run(var.assign(value))  # attempt to add sumamry without assignment will raise error
                self.log_string += ' {0}-{1}: {2:>0.4f}'.format(log_type.title(), var.name.split(':')[0], value)
            else:  # ex) tf.placeholder, tf.constant
                images_dict[var] = value

        self.current_step = tf.train.global_step(sess, tf.train.get_global_step())
        summary = sess.run(tf.summary.merge_all(), feed_dict=images_dict)  # None (by default given no image)

        if log_type == 'train':
            self.summary_writer_train.add_summary(summary, self.current_step)
            self.summary_writer_train.add_graph(sess.graph, self.current_step)

        else:
            self.summary_writer_eval.add_summary(summary, self.current_step)
            self.summary_writer_eval.add_graph(sess.graph, self.current_step)

    def display_summary(self, time_stamp=False):
        if time_stamp:  # total elapsed time after training started
            time_elapsed = str(datetime.datetime.now() - self.start_time)
            self.log_string += ' Time Elasped: {0}'.format(time_elapsed.split('.')[0])
        print('Step: {0:>4d}{1}'.format(self.current_step, self.log_string), flush=True)
        self.log_string = ''  # initialize