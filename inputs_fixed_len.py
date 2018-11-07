import os
import pickle
import random
import tensorflow as tf
import numpy as np

def make_tf_record_file(data, data_type):

    def make_example(timeseries_data, y_list):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        timeseries_data = timeseries_data.astype(np.float32)
        timeseries_maxlen =  timeseries_data.shape[0]
        timeseries_data_bytes = tf.compat.as_bytes(timeseries_data.values.tostring())

        features = tf.train.Features(feature={'x':_bytes_feature([timeseries_data_bytes]),
                                              'y': _int64_feature(y_list),
                                              'max_len': _int64_feature([timeseries_maxlen])
                                              })

        example = tf.train.Example(features = features)

        return example


    writer = tf.python_io.TFRecordWriter(
        'data/tfrecord_data/{}.tfrecord'.format(data_type))

    for key, val in data.items():
        timeseries_data, static_data, kl_grade = val

        kl_grade_list = [kl_grade['Rt'], kl_grade['Lt']]

        ex = make_example(timeseries_data, kl_grade_list)
        writer.write(ex.SerializeToString())
    writer.close()
    print('tfrecord {} made'.format(data_type))


def make_tfrecord_data(train_test_split_ratio):

    data_file = 'data/preprocessed_data.pkl'
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    key_list = list(data.keys())
    random.shuffle(key_list)

    train_len = int(len(key_list) * train_test_split_ratio)
    train_key_list = key_list[:train_len]
    test_key_list = key_list[train_len:]

    train_data = {k:v for k,v in data.items() if k in train_key_list}
    test_data = {k: v for k, v in data.items() if k in test_key_list}

    make_tf_record_file(train_data, 'train')
    make_tf_record_file(test_data, 'test')



def inputs(batch_size, train_test_split_ratio, num_parallel_calls=10):

    def decode(serialized_example):
        """Parses an image and label from the given `serialized_example`."""

        parsed = tf.parse_single_example(
            serialized_example,
            features={'x': tf.FixedLenFeature([], tf.string),
                      'y': tf.FixedLenFeature([2], tf.int64),
                      'max_len': tf.FixedLenFeature([], tf.int64)
        })

        x = tf.decode_raw(parsed['x'], tf.float32)
        x = tf.reshape(x, [101, 46])

        # x = tf.cast(x, tf.float32)

        y = tf.cast(parsed['y'], tf.int32)

        max_len = tf.cast(parsed['max_len'], tf.int32)

        return {'x': x, 'y': y, 'max_len': max_len}

    def make_dataset(file_list, data_type):
        dataset = tf.data.TFRecordDataset(file_list)

        dataset = dataset.map(decode, num_parallel_calls=num_parallel_calls)

        if data_type == 'train':
            dataset = dataset.shuffle(buffer_size = batch_size * 10)
        dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(batch_size * 10)

        return dataset

    tfrecord_data_dir = 'data/tfrecord_data'

    if not os.path.exists(tfrecord_data_dir):
        os.makedirs(tfrecord_data_dir)
        make_tfrecord_data(train_test_split_ratio)

    trn_dataset = make_dataset('{}/train.tfrecord'.format(tfrecord_data_dir), 'train')

    test_dataset = make_dataset('{}/test.tfrecord'.format(tfrecord_data_dir), 'test')

    #

    iterator = tf.data.Iterator.from_structure(trn_dataset.output_types, trn_dataset.output_shapes)

    next_batch = iterator.get_next()

    trn_init_op = iterator.make_initializer(trn_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    return next_batch, trn_init_op, test_init_op


def test():
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        next_batch, trn_init_op, test_init_op = inputs(32, 0.1)

        sess.run(trn_init_op)

        a = sess.run(next_batch)

        print(a['x'].shape)

if __name__ =='__main__':
    test()