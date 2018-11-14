import tensorflow as tf


class Model():

    def __init__(self, inputs, avg_pool, selected_cols, rnn_hidden_dim, conv_layers,
                 mlp_layers, y_size):

        self.is_training = tf.placeholder(tf.bool, shape=None)
        tf.add_to_collection('is_training', self.is_training)

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.epoch = tf.Variable(0, trainable=False, name='epoch')

        self.base_learning_rate = 1e-4
        self.batch_size_for_learning_rate = 32

        def build_mlp(inputs, layers, drop_out=None):
            for layer_num, layer_dim in enumerate(layers):
                with tf.variable_scope('dense_{}'.format(layer_num)):
                    inputs = tf.layers.dense(inputs, layer_dim, activation=tf.nn.relu)
                    if drop_out == layer_num:
                        inputs = tf.layers.dropout(inputs, rate=0.5,
                                                   training=self.is_training)
                        print('dropout')
                    print('mlp', inputs.shape)
            return inputs

        def build_conv1d(inputs, layers):
            for layer_num, layer_config in enumerate(layers):
                filters, kernel_size, strides = layer_config
                with tf.variable_scope('conv1d_{}'.format(layer_num)):
                    inputs = tf.layers.conv1d(inputs, filters, kernel_size, strides,
                                              padding='same')
                    # inputs = tf.layers.batch_normalization(inputs,
                    #                                        training=self.is_training)
                    inputs = tf.nn.relu(inputs)
                    print('conv', inputs.shape)
                    inputs = tf.layers.max_pooling1d(inputs, pool_size=2, strides=2)
                    print('conv maxpool', inputs.shape)
            return inputs

        x = inputs['x']

        print('input', x.shape)

        if avg_pool:
            with tf.variable_scope('average_pooling_1d'):
                pool_size, strides = avg_pool
                x = tf.layers.average_pooling1d(x, pool_size, strides)
                print('input pool', x.shape)

        import numpy as np

        batch_size = tf.shape(x)[0]

        y = inputs['y']
        if y_size == 2:
            y = tf.cast(tf.greater_equal(y, 2),tf.int32)

        y_r = y[:, 0]
        y_l = y[:, 1]

        max_len = inputs['max_len']

        with tf.variable_scope('conv_layer'):
            print('conv')
            conv_outputs = build_conv1d(x, conv_layers)

            # last_states = tf.reduce_mean(conv_outputs, axis=1)

            last_states = tf.layers.flatten(conv_outputs)
            print('conv output flatten', last_states.shape)

        # with tf.variable_scope('rnn_layer'):
        #     rnn_cell = tf.contrib.rnn.GRUCell(num_units=rnn_hidden_dim)
        #     rnn_outputs, last_states = tf.nn.dynamic_rnn(
        #         cell=rnn_cell,
        #         inputs=x, #[bs, time_len, channel]
        #         dtype=tf.float32,
        #         # parallel_iterations=64,
        #         # sequence_length=tf.ones(batch_size) * 101
        #     )
        #
        #     # https://github.com/tensorflow/tensorflow/issues/19568 update_ops crashses
        #     # when rnn length is 32
        #     # parallel_iterations = 71
        #     # sequence_length = qst_len,
        #     # GRU

        with tf.variable_scope('mlp_right'):
            print('mlp_right')
            mlp_r_output = build_mlp(last_states, mlp_layers)
            logits_r = tf.layers.dense(mlp_r_output,
                                       y_size,
                                       use_bias=False)

        with tf.variable_scope('mlp_left'):
            print('mlp_left')
            mlp_l_output = build_mlp(last_states, mlp_layers)
            logits_l = tf.layers.dense(mlp_l_output,
                                       y_size,
                                       use_bias=False)

        with tf.variable_scope('loss'):
            '''Left

            4    133
            0     99
            3     90
            1     68
            2     61

            Right
            4    165
            0    100
            3     72
            2     71
            1     43            
            '''

            if y_size == 2:
                xent_loss_raw_r = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_r, logits=logits_r)
                xent_loss_raw_l = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_l, logits=logits_l)
            else:
                l_w = tf.constant([1.0, 2.0, 2.0, 2.0, 1.0])
                r_w = tf.constant([1.0, 3.0, 2.0, 2.0, 1.0])

                # weights_r = tf.gather(l_w, y_r)
                # weights_l = tf.gather(r_w, y_l)

                y_r_onehot = tf.one_hot(y_r, y_size, dtype=tf.float32)
                y_l_onehot = tf.one_hot(y_l, y_size, dtype=tf.float32)

                weights_r = tf.reduce_sum(l_w * y_l_onehot, axis=1)
                weights_l = tf.reduce_sum(r_w * y_r_onehot, axis=1)

                xent_loss_raw_r = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=y_r_onehot, logits=logits_r)
                xent_loss_raw_l = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=y_l_onehot, logits=logits_l)

                xent_loss_raw_r = xent_loss_raw_r * weights_r
                xent_loss_raw_l = xent_loss_raw_l * weights_l




            xent_loss_raw = xent_loss_raw_r + xent_loss_raw_l
            xent_loss_raw = tf.check_numerics(xent_loss_raw,
                                              'nan value found in loss raw')
            self.loss = tf.reduce_mean(xent_loss_raw)

        with tf.variable_scope('learning_rate'):

            self.increment_epoch_op = tf.assign(self.epoch, self.epoch + 1)

            if self.batch_size_for_learning_rate < 64:
                self.learning_rate = self.base_learning_rate
            else:
                self.learning_rate = tf.train.polynomial_decay(
                    self.base_learning_rate,
                    self.epoch,
                    decay_steps=5,
                    end_learning_rate=self.base_learning_rate * (
                                self.batch_size_for_learning_rate / 64),
                )

        with tf.variable_scope('summary'):

            pred_r = tf.argmax(logits_r, axis=1)
            pred_r = tf.cast(pred_r, tf.int32)
            pred_l = tf.argmax(logits_l, axis=1)
            pred_l = tf.cast(pred_l, tf.int32)
            self.prediction = tf.stack([pred_r, pred_l], axis=1)
            tf.add_to_collection('pred', self.prediction)

            self.accuracy, _ = tf.metrics.accuracy(y,
                                                   self.prediction,
                                                   updates_collections='summary_update')

            summary_trn = list()
            summary_trn.append(tf.summary.scalar('trn_accuracy', self.accuracy))
            summary_trn.append(tf.summary.scalar('learning_rate', self.learning_rate))

            summary_test = list()
            summary_test.append(tf.summary.scalar('test_accuracy', self.accuracy))

            for leg_idx, leg in zip(range(2), 'rl'):
                for y_val in range(y_size):
                    acc_type_mask = tf.equal(y_val, y[:, leg_idx])
                    ans_tmp = tf.boolean_mask(y[:, leg_idx], acc_type_mask)
                    pred_tmp = tf.boolean_mask(self.prediction[:, leg_idx], acc_type_mask)
                    acc_tmp, _ = tf.metrics.accuracy(ans_tmp, pred_tmp,
                                                     updates_collections='summary_update')
                    summary_trn.append(tf.summary.scalar('trn_rec_{}_{}_acc'.format(leg,
                                                                                    y_val),
                                                         acc_tmp))
                    summary_test.append(tf.summary.scalar('test_rec_{}_{}_acc'.format(leg,
                                                                                      y_val),
                                                          acc_tmp))

            for leg_idx, leg in zip(range(2), 'rl'):
                for y_val in range(y_size):
                    acc_type_mask = tf.equal(y_val, self.prediction[:, leg_idx])
                    ans_tmp = tf.boolean_mask(y[:, leg_idx], acc_type_mask)
                    pred_tmp = tf.boolean_mask(self.prediction[:, leg_idx], acc_type_mask)
                    acc_tmp, _ = tf.metrics.accuracy(ans_tmp, pred_tmp,
                                                     updates_collections='summary_update')
                    summary_trn.append(tf.summary.scalar('trn_prec_{}_{}_acc'.format(leg,
                                                                                     y_val),
                                                         acc_tmp))
                    summary_test.append(tf.summary.scalar('test_prec_{}_{}_acc'.format(leg,
                                                                                       y_val),
                                                          acc_tmp))

            self.summary_trn = tf.summary.merge(summary_trn)

            self.summary_test = tf.summary.merge(summary_test)

            with tf.variable_scope('running_summary'):

                trn_running_summary = [tf.summary.scalar('trn_xent_loss', self.loss)]

                test_running_summary = [tf.summary.scalar('test_xent_loss', self.loss)]

                input_gradients = tf.reduce_sum(tf.abs(tf.gradients(self.loss, x)), axis=1)

                for idx, col in enumerate(selected_cols):
                    summary_value = input_gradients[:, idx]
                    trn_running_summary.append(tf.summary.histogram('trn_{}'.format(col),
                                                                    summary_value))
                    test_running_summary.append(tf.summary.histogram('test_{}'.format(col),
                                                                     summary_value))

                self.trn_running_summary = tf.summary.merge(trn_running_summary)

                self.test_running_summary = tf.summary.merge(test_running_summary)

        with tf.variable_scope('train'):

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.summary_update_ops = tf.get_collection('summary_update')

            # self.summary_update_ops
            with tf.control_dependencies(self.update_ops):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    self.loss, global_step=self.global_step)