import os
import pickle
import random
import tensorflow as tf
import numpy as np

def make_tf_record_file(data, selected_cols_idx, tfrecord_data_dir, data_type):

    def make_example(timeseries_data, y_list):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        timeseries_data = timeseries_data.iloc[:,selected_cols_idx]
        timeseries_data = timeseries_data.astype(np.float32)
        timeseries_maxlen =  timeseries_data.shape[0]
        timeseries_data_bytes = tf.compat.as_bytes(timeseries_data.values.tostring())

        features = tf.train.Features(feature={'x':_bytes_feature([timeseries_data_bytes]),
                                              'y': _int64_feature(y_list),
                                              'max_len': _int64_feature([timeseries_maxlen])
                                              })

        example = tf.train.Example(features = features)

        return example


    writer = tf.python_io.TFRecordWriter('{}/{}.tfrecord'.format(tfrecord_data_dir, data_type))

    for key, val in data.items():

        timeseries_data, static_data, kl_grade = val
        sorted_cols = sorted(timeseries_data.columns)
        timeseries_data = timeseries_data[sorted_cols]

        kl_grade_list = [kl_grade['Rt'], kl_grade['Lt']]

        ex = make_example(timeseries_data, kl_grade_list)
        writer.write(ex.SerializeToString())
    writer.close()
    print('tfrecord {} made'.format(data_type))


def make_tfrecord_data(train_test_split_ratio, selected_cols_idx, tfrecord_data_dir):

    data_file = 'data/preprocessed_data.pkl'
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    key_list = list(data.keys())
    random.shuffle(key_list)

    train_len = int(len(key_list) * train_test_split_ratio)
    train_key_list = key_list[:train_len]
    test_key_list = key_list[train_len:]

    # from sklearn.model_selection import StratifiedShuffleSplit
    # key_list = list()
    # label_list = list()
    # for key, val in data.items():
    #     key_list.append(key)
    #     label_list.append(val[])
    # sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2)
    # train_idx = range(len(key_list))
    # sss.split(train_idx, )

    train_data = {k:v for k,v in data.items() if k in train_key_list}
    test_data = {k: v for k, v in data.items() if k in test_key_list}

    make_tf_record_file(train_data, selected_cols_idx, tfrecord_data_dir, 'train')
    make_tf_record_file(test_data, selected_cols_idx, tfrecord_data_dir, 'test')



def inputs(batch_size, train_test_split_ratio, selected_cols_idx,
           col_str, num_parallel_calls=10):

    num_var = len(selected_cols_idx)

    def decode(serialized_example):
        """Parses an image and label from the given `serialized_example`."""

        parsed = tf.parse_single_example(
            serialized_example,
            features={'x': tf.FixedLenFeature([], tf.string),
                      'y': tf.FixedLenFeature([2], tf.int64),
                      'max_len': tf.FixedLenFeature([], tf.int64,)
        })

        x = tf.decode_raw(parsed['x'], tf.float32)
        x = tf.reshape(x, [101, num_var])

        y = tf.cast(parsed['y'], tf.int32)

        max_len = tf.cast(parsed['max_len'], tf.int32)

        return {'x': x, 'y': y, 'max_len': max_len}

    def make_dataset(file_list, data_type):
        dataset = tf.data.TFRecordDataset(file_list)

        dataset = dataset.map(decode, num_parallel_calls=num_parallel_calls)

        # dataset = dataset.filter(lambda x: tf.equal(x['y'][0], x['y'][1]))

        if data_type == 'train':
            dataset = dataset.shuffle(buffer_size = batch_size * 10)
        dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(batch_size * 10)

        return dataset

    tfrecord_data_dir = 'data/tfrecord_data/{}'.format(col_str)

    if not os.path.exists(tfrecord_data_dir):
        os.makedirs(tfrecord_data_dir)
        make_tfrecord_data(train_test_split_ratio, selected_cols_idx, tfrecord_data_dir)

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

        pooled_x = tf.layers.average_pooling1d(next_batch['x'], 2, strides=2)

        sess.run(trn_init_op)

        x, y, pooled_x_val = sess.run([next_batch['x'],
                                    next_batch['y'],
                                    pooled_x])

        import matplotlib.pyplot as plt
        for i in range(32):
            print(y[i])
            plt.plot(x[i, :, 0])
            plt.plot(pooled_x_val[i, :, 0])
            plt.show()

        print(a['x'].shape)

if __name__ =='__main__':
    test()