import os
import pickle
import random
import tensorflow as tf
import numpy as np

def make_tf_record_file(data, timeseries_cols, static_cols, tfrecord_data_dir,
                        data_type):

    def make_example(timeseries_data, static_data, timeseries_data_agg, y_list):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        timeseries_data = timeseries_data.loc[:, timeseries_cols]
        timeseries_data = timeseries_data.astype(np.float32)
        # timeseries_maxlen =  timeseries_data.shape[0]
        timeseries_data_bytes = tf.compat.as_bytes(timeseries_data.values.tostring())

        static_data = static_data['value'][list(static_cols)] # tuples are recognized as
        #  composite keys(하나하나씩이 아니라, 그 전체 조합이 하나의 Key로), but if default args is in list,
        # because it is mutable,
        # antoher error occurs


        # [x for x in data_dict[list(data_dict.keys())[0]][3].index if 'Hip Flexion Angle' in x]

        static_data = static_data.astype(np.float32)
        static_data_bytes = tf.compat.as_bytes(static_data.values.tostring())

        agg_selected_cols_idx = list()
        for col in timeseries_cols:
            agg_selected_cols_idx.append(col + '_auc')
            agg_selected_cols_idx.append(col + '_mean')
            agg_selected_cols_idx.append(col + '_rms')
            agg_selected_cols_idx.append(col + '_kurtosis')

        timeseries_data_agg = timeseries_data_agg[agg_selected_cols_idx]

        timeseries_data_agg = timeseries_data_agg.astype(np.float32)

        timeseries_data_agg = tf.compat.as_bytes(timeseries_data_agg.values.tostring())

        features = tf.train.Features(feature={'x_t':_bytes_feature([timeseries_data_bytes]),
                                              'x_s': _bytes_feature([static_data_bytes]),
                                              'x_t_agg': _bytes_feature([timeseries_data_agg]),
                                              'y': _int64_feature(y_list)
                                              })

        example = tf.train.Example(features = features)

        return example


    writer = tf.python_io.TFRecordWriter('{}/{}.tfrecord'.format(tfrecord_data_dir, data_type))

    for key, val in data.items():

        timeseries_data, static_data, kl_grade, timeseries_data_agg = val
        sorted_cols = sorted(timeseries_data.columns, key=lambda x:(x[-1]=='r', x))
        timeseries_data = timeseries_data[sorted_cols]

        kl_grade_list = [kl_grade['Rt'], kl_grade['Lt']]

        ex = make_example(timeseries_data, static_data, timeseries_data_agg, kl_grade_list)
        writer.write(ex.SerializeToString())
    writer.close()
    print('tfrecord {} made'.format(data_type))


def make_tfrecord_data(train_test_split_ratio, timeseries_cols,
                       static_cols, tfrecord_data_dir):

    # data_file = 'data/preprocessed_data.pkl'
    data_file = 'data/preprocessed_data_hip_extension_0_deleted.pkl'
    print('hip extension moment 0 data_deleted' * 100)

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

    make_tf_record_file(train_data, timeseries_cols, static_cols, tfrecord_data_dir,
                        'train')
    make_tf_record_file(test_data, timeseries_cols, static_cols, tfrecord_data_dir,
                        'test')



def inputs(batch_size, train_test_split_ratio, timeseries_cols,
           col_str, static_cols =('Age', 'Height', 'Weight', 'R_Speed'),
           num_parallel_calls=10):

    num_var = len(timeseries_cols)

    def decode(serialized_example):
        """Parses an image and label from the given `serialized_example`."""

        parsed = tf.parse_single_example(
            serialized_example,
            features={'x_t': tf.FixedLenFeature([], tf.string),
                      'x_s': tf.FixedLenFeature([], tf.string),
                      'x_t_agg': tf.FixedLenFeature([], tf.string),
                      'y': tf.FixedLenFeature([2], tf.int64)
        })

        x_t = tf.decode_raw(parsed['x_t'], tf.float32)
        # x_t.set_shape([101, num_var])
        x_t = tf.reshape(x_t, [101, num_var])

        x_t_agg = tf.decode_raw(parsed['x_t_agg'], tf.float32)
        x_t_agg.set_shape(num_var * 4) # 4 [_mean, _auc, _rms, _kurtosis]

        x_s = tf.decode_raw(parsed['x_s'], tf.float32)
        x_s.set_shape(len(static_cols))

        y = tf.cast(parsed['y'], tf.int32)

        return {'x_t': x_t, 'x_t_agg': x_t_agg, 'x_s':x_s, 'y': y}

    def make_dataset(file_list, data_type):
        dataset = tf.data.TFRecordDataset(file_list)

        dataset = dataset.map(decode, num_parallel_calls=num_parallel_calls)

        # dataset = dataset.filter(lambda x: tf.equal(x['y'][0], x['y'][1]))
        # print('filtered only data with same kl degree' * 20)

        if data_type == 'train':
            dataset = dataset.shuffle(buffer_size = batch_size * 10)
        dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(batch_size * 10)

        return dataset

    tfrecord_data_dir = 'data/tfrecord_data/{}'.format(col_str)

    if not os.path.exists(tfrecord_data_dir):
        os.makedirs(tfrecord_data_dir)
        make_tfrecord_data(train_test_split_ratio, timeseries_cols, static_cols,
                           tfrecord_data_dir)

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



if __name__ =='__main__':
    test()