import tensorflow as tf
import os
import pickle
import random

def make_seq_tf_record_file(data, data_type):

    def make_example(timeseries_data, y):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        feature_list = dict()
        for col in timeseries_data:
            feature = [_float_feature(x) for x in timeseries_data[col]]
            feature_list[col] = tf.train.FeatureList(feature=feature)

        feature_lists = tf.train.FeatureLists(feature_list=feature_list)


        # question_features = [_int64_feature([x]) for x in qst]
        #
        # feature_list = {'question': tf.train.FeatureList(feature=question_features)}
        #
        # feature_lists = tf.train.FeatureLists(feature_list=feature_list)

        timeseries_maxlen =  timeseries_data.shape[0]

        context_features = tf.train.Features(
            feature={
                'y': _int64_feature([y]),
                'len': _int64_feature([timeseries_maxlen])})

        # context_features = tf.train.Features(feature={'img_raw':_bytes_feature([img]),
        #                                               'answer': _int64_feature([answer]),
        #                                               'question_len':_int64_feature([len(qst)])
        # })

        example = tf.train.SequenceExample(feature_lists=feature_lists,
                                           context=context_features)
        return example


    writer = tf.python_io.TFRecordWriter(
        'data/tfrecord_data/{}.tfrecord'.format(data_type))

    for key, val in data.items():
        timeseries_data, static_data, kl_grade = val
        ex = make_example(timeseries_data, kl_grade)
        writer.write(ex.SerializeToString())
    writer.close()
    print('tfrecord {} made'.format(data_type))


def make_tfrecord_data(train_test_split_ratio):

    if not os.path.exists('data/tfrecord_data'):
        os.makedirs('data/tfrecord_data')

    data_file = 'data/preprocessed_data/preprocessed_data.pkl'
    with open(data_file) as f:
        data = pickle.load(f)

    key_list = list(data.keys())
    random.shuffle(key_list)

    train_len = int(len(key_list) * train_test_split_ratio)
    train_key_list = key_list[:train_len]
    test_key_list = key_list[train_len:]

    train_data = {k:v for k,v in data if k in train_key_list}
    test_data = {k: v for k, v in data if k in test_key_list}

    make_seq_tf_record_file(train_data, 'train')
    make_seq_tf_record_file(test_data, 'test')
    return next_batch, trn_init_op, test_init_op