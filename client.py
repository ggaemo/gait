import inputs_fixed_len
import tensorflow as tf
import argparse
import os
import model
import re
import time

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-learning_rate', type=float, default=1e-4)
parser.add_argument('-num_epochs', type=int, default=500)
parser.add_argument('-train_test_split_ratio', type=float, default=0.8)
parser.add_argument('-rnn_hidden_dim', type=int)
parser.add_argument('-f_phi_layers', type=int, nargs='+')
parser.add_argument('-option', type=str, default='')
parser.add_argument('-restore', action='store_true', default=False)
parser.add_argument('-save_interval', type=int, default=10)
args = parser.parse_args()


def layer_config_to_str(layer_config):
    return '-'.join([str(x) for x in layer_config])

dir_format = 'model/bs-{}_rnn-{}-f_phi-{}'
model_dir = dir_format.format(
    args.batch_size,
    args.rnn_hidden_dim,
    layer_config_to_str(args.f_phi_layers)
)

if args.option:
    model_dir = '{}_option-{}'.format(model_dir, args.option)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
elif 'checkpoint' in os.listdir(model_dir) and not args.restore:
    print('saved model exists')
    raise FileExistsError


with tf.Graph().as_default():

    with tf.variable_scope('inputs'):
        next_batch, trn_init_op, test_init_op = inputs_fixed_len.inputs(
            args.batch_size,
            args.train_test_split_ratio,
            num_parallel_calls=20)
        tf.add_to_collection('test_init_op', test_init_op)
        tf.add_to_collection('train_init_op', trn_init_op)

    with tf.variable_scope('model'):
        model = model.Model(next_batch, args.rnn_hidden_dim, args.f_phi_layers, 5)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=5)
        summary_writer = tf.summary.FileWriter(model_dir, flush_secs=5, graph=sess.graph)
        global_step = 1
        if args.restore:
            latest_model = tf.train.latest_checkpoint(model_dir)
            print('restored model from ', latest_model)
            epoch_num = int(re.search('model.ckpt-(\d+)', latest_model).group(1))
            sess.run(tf.assign(model.epoch, epoch_num))
            saver.restore(sess, latest_model)
        else:
            sess.run(tf.global_variables_initializer())
            epoch_num = sess.run(model.epoch)

        for _ in range(args.num_epochs):
            print('epoch num', epoch_num, 'batch iteration', global_step)
            prev = time.time()
            sess.run(trn_init_op)
            sess.run(tf.local_variables_initializer())

            trn_feed = {model.is_training: True}

            try:
                while True:

                    if global_step % args.save_interval == 0:
                        _, global_step, trn_loss_summary, _ = sess.run([model.train_op,
                                                                     model.global_step,
                                                                     model.trn_loss_summary,
                                                                     model.summary_update_ops
                                                                     ],
                                                                    trn_feed
                                                                    )

                        summary_writer.add_summary(trn_loss_summary, epoch_num)
                    else:
                        _, global_step, loss, _ = sess.run([model.train_op,
                                                   model.global_step,
                                                   model.loss,
                                                         model.summary_update_ops
                                                   ],
                                                  trn_feed
                                                  )


            except tf.errors.OutOfRangeError:
                sess.run(model.increment_epoch_op)
                epoch_num = sess.run(model.epoch)
                print('out of range', 'epoch', epoch_num, 'iter', global_step)
                now = time.time()
                summary_value, trn_acc = sess.run([model.summary_trn,
                                                   model.accuracy],
                                                  {model.is_training: False})
                summary_writer.add_summary(summary_value, global_step=epoch_num)

                sess.run(test_init_op)
                sess.run(tf.local_variables_initializer())  # metrics value init to 0

                try:
                    print('test_start')
                    tmp_step = 0

                    while True:
                        if tmp_step % args.save_interval == 0:
                            _, test_loss_summary = sess.run([model.summary_update_ops,
                                                             model.test_loss_summary],
                                                            {model.is_training: False})
                            summary_writer.add_summary(test_loss_summary,
                                                       global_step=epoch_num)
                        else:
                            sess.run(model.summary_update_ops, {model.is_training: False})

                        tmp_step += 1

                except tf.errors.OutOfRangeError:
                    print('test_start end')
                    summary_value, test_acc = sess.run([model.summary_test,
                                                        model.accuracy],
                                                       {
                                                           model.is_training: False})
                    summary_writer.add_summary(summary_value, global_step=epoch_num)

                minutes = (now - prev) / 60
                result = 'num iter: {} | trn_acc : {} test acc : {}'.format(
                    global_step, trn_acc, test_acc)

                message = 'took {} min'.format(minutes)
                print(model_dir)
                print(result)
                print(message)

                saver.save(sess, os.path.join(model_dir, 'model.ckpt'),
                           global_step=epoch_num)

