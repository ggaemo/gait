import inputs_fixed_len
import tensorflow as tf
import argparse
import os
import model
import re
import time
import numpy as np
import utils
import matplotlib.pyplot as plt

import sklearn.ensemble
import sklearn.metrics
import pickle
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str)
parser.add_argument('-train_test_split_ratio', type=float, default=0.8)
parser.add_argument('-timeseries_cols', type=str, nargs='+')
args = parser.parse_args()

if args.model == 'rf':
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, max_depth=5,
                                                    min_samples_leaf=6,
                                                    class_weight='balanced')
elif args.model == 'svm':
    import sklearn.svm
    model = sklearn.svm.SVC(decision_function_shape='ovo')

if args.timeseries_cols:
    timeseries_cols = [x for x in utils.columns if any(y.lower() in x.lower() for y in
                                                     args.timeseries_cols)]
else:
    timeseries_cols = ['Knee adduction moment_l',
                     'Knee adduction moment_r',
                     'Knee Flexion Angle_l',
                     'Knee Flexion Angle_r',
                     'Hip abduction moment_l',
                     'Hip abduction moment_r',
                     'Hip extension moment_l',
                     'Hip extension moment_r']

print(timeseries_cols)

timeseries_cols_wo_leg = sorted(set([x.split('_')[0] for x in timeseries_cols]))
timeseries_cols_to_str = '_'.join([
    ''.join([y[0] for y in x.split()])
    for x in timeseries_cols_wo_leg])

timeseries_cols = sorted(timeseries_cols)

npz_dir = lambda x: 'data/tfrecord_data/{}/{}.pkl'.format(timeseries_cols_to_str, x)
if not os.path.exists(npz_dir('train')):
    with tf.Graph().as_default():
        with tf.variable_scope('inputs'):
            next_batch, trn_init_op, test_init_op = inputs_fixed_len.inputs(
                9999,
                args.train_test_split_ratio,
                timeseries_cols,
                timeseries_cols_to_str)
        with tf.Session() as sess:
            sess.run(trn_init_op)
            train_data = sess.run(next_batch)
            sess.run(test_init_op)
            test_data = sess.run(next_batch)
    with open(npz_dir('train'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(npz_dir('test'), 'wb') as f:
        pickle.dump(test_data, f)

else:
    with open(npz_dir('train'), 'rb') as f:
        train_data = pickle.load(f)
    with open(npz_dir('test'), 'rb') as f:
        test_data = pickle.load(f)


train_x = np.concatenate((train_data['x_t_agg'], train_data['x_s']), axis=1)
train_y = train_data['y']

test_x = np.concatenate((test_data['x_t_agg'], test_data['x_s']), axis=1)
test_y = test_data['y']

model.fit(train_x, train_y)

train_pred = model.predict(train_x)
test_pred = model.predict(test_x)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def get_accuracy(y, pred):
    for i, leg in zip(range(2), 'rl'):
        tmp_y = y[:, i]
        tmp_pred = pred[:, i]
        acc = sklearn.metrics.accuracy_score(tmp_y, tmp_pred)
        cnf_matrix = sklearn.metrics.confusion_matrix(tmp_y, tmp_pred)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, ['0', '1', '2', '3', '4'], True,
                              timeseries_cols_to_str+'_'+leg)
        plt.show()
        print(leg, acc)

get_accuracy(train_y, train_pred)
get_accuracy(test_y, test_pred)
if args.model == 'rf':
    print(model.feature_importances_)
# model.predict(test_data['x_t_agg'])
#
#
#
#
#
#         for _ in range(args.num_epochs):
#             print('epoch num', epoch_num, 'batch iteration', global_step)
#             prev = time.time()
#             sess.run(trn_init_op)
#             sess.run(tf.local_variables_initializer())
#
#             trn_feed = {model.is_training: True}
#
#             a = sess.run(model.get)
#
#             try:
#                 while True:
#                     if global_step % args.save_interval == 0:
#                         _, global_step, trn_loss_summary, _ = sess.run([model.train_op,
#                                                                      model.global_step,
#                                                                      model.trn_running_summary,
#                                                                      model.summary_update_ops
#                                                                      ],
#                                                                     trn_feed
#                                                                     )
#
#                         summary_writer.add_summary(trn_loss_summary, epoch_num)
#                     else:
#                         _, global_step, loss, _ = sess.run([model.train_op,
#                                                             model.global_step,
#                                                             model.loss,
#                                                             model.summary_update_ops
#                                                             ],
#                                                            trn_feed
#                                                            )
#
#             except tf.errors.OutOfRangeError:
#                 sess.run(model.increment_epoch_op)
#                 epoch_num = sess.run(model.epoch)
#                 print('out of range', 'epoch', epoch_num, 'iter', global_step)
#                 now = time.time()
#                 summary_value, trn_acc = sess.run([model.summary_trn,
#                                                    model.accuracy],
#                                                   {model.is_training: False})
#                 summary_writer.add_summary(summary_value, global_step=epoch_num)
#
#                 sess.run(test_init_op)
#                 sess.run(tf.local_variables_initializer())  # metrics value init to 0
#
#                 try:
#                     print('test_start')
#                     tmp_step = 0
#
#                     while True:
#                         if tmp_step % args.save_interval == 0:
#                             _, test_loss_summary = sess.run([model.summary_update_ops,
#                                                              model.test_running_summary],
#                                                             {model.is_training: False})
#                             summary_writer.add_summary(test_loss_summary,
#                                                        global_step=epoch_num)
#                         else:
#                             sess.run(model.summary_update_ops, {model.is_training: False})
#
#                         tmp_step += 1
#
#                 except tf.errors.OutOfRangeError:
#                     print('test_start end')
#                     summary_value, test_acc = sess.run([model.summary_test,
#                                                         model.accuracy],
#                                                        {
#                                                            model.is_training: False})
#                     summary_writer.add_summary(summary_value, global_step=epoch_num)
#
#                 minutes = (now - prev) / 60
#                 result = 'num iter: {} | trn_acc : {} test acc : {}'.format(
#                     global_step, trn_acc, test_acc)
#
#                 message = 'took {} min'.format(minutes)
#                 print(model_dir)
#                 print(result)
#                 print(message)
#
#                 saver.save(sess, os.path.join(model_dir, 'model.ckpt'),
#                            global_step=epoch_num)
#
