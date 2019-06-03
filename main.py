# coding: utf8
"""
source code for the paper titled "Exploiting Stereo Sound Channels to Boost Performance of Neural Network-Based Music
Transcription"

References:
[1] Curtis Hawthorne, Erich Elsen, Jialin Song, Adam Roberts, Ian Simon, Colin Raffel, Jesse Engel, Sageev Oore,
    and Douglas Eck, “Onsets and frames: Dual- objective piano transcription,” in Proceedings of the 19th International
    Society for Music Information Retrieval Conference, ISMIR 2018.
[2] Rainer Kelz, Matthias Dorfer, Filip Korzeniowski, Sebastian Bock, Andreas Arzt, and Gerhard Widmer,
    “On the potential of simple framewise approaches to piano transcription,” in Proceedings of
    the 17th International Society for Music Information Retrieval Conference, ISMIR 2016
"""

from __future__ import print_function

DEBUG = True  # in debug mode the numbers of recordings are minimized for fast debugging
GPU_ID = 0  # in case you have multiple GPUs, select the one to run this script

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import glob
import re
import librosa
import librosa.display
import numpy as np
from argparse import Namespace
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import datetime
from magenta.common import flatten_maybe_padded_sequences
import madmom
import magenta.music
import collections


# contain the common, miscellaneous functions
class MiscFns(object):
    """Miscellaneous functions"""

    @staticmethod
    def filename_to_id(filename):
        """Translate a .wav or .mid path to a MAPS sequence id.
        This snippet is from [1]
        """
        return re.match(r'.*MUS-(.+)_[^_]+\.\w{3}',
                        os.path.basename(filename)).group(1)

    @staticmethod
    def log_filter_bank_fn():
        """
        generate a logarithmic filterbank
        """
        log_filter_bank_basis = madmom.audio.filters.LogarithmicFilterbank(
            bin_frequencies=librosa.fft_frequencies(sr=16000, n_fft=2048),
            num_bands=48,
            fmin=librosa.midi_to_hz([27])[0],
            fmax=librosa.midi_to_hz([114])[0] * 2. ** (1. / 48)
        )
        log_filter_bank_basis = np.array(log_filter_bank_basis)
        assert log_filter_bank_basis.shape[1] == 229
        assert np.abs(np.sum(log_filter_bank_basis[:, 0]) - 1.) < 1e-3
        assert np.abs(np.sum(log_filter_bank_basis[:, -1]) - 1.) < 1e-3

        return log_filter_bank_basis

    @staticmethod
    def spectrogram_fn(samples, log_filter_bank_basis, spec_stride):
        """
        generate spectrogram
        """
        num_frames = (len(samples) - 1) // spec_stride + 1
        stft = librosa.stft(y=samples, n_fft=2048, hop_length=spec_stride)
        assert num_frames <= stft.shape[1] <= num_frames + 1
        if stft.shape[1] == num_frames + 1:
            stft = stft[:, :num_frames]
        stft = stft / 1024
        stft = np.abs(stft)
        stft = 20 * np.log10(stft + 1e-7) + 140
        lm_mag = np.dot(stft.T, log_filter_bank_basis)
        assert lm_mag.shape[1] == 229
        lm_mag = np.require(lm_mag, dtype=np.float32, requirements=['C'])

        return lm_mag

    @staticmethod
    def times_to_frames_fn(spec_stride, start_time, end_time):
        """
        convert time to frame
        """
        assert spec_stride & 1 == 0
        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)
        start_frame = (start_sample + spec_stride // 2) // spec_stride
        end_frame = (end_sample + spec_stride // 2 - 1) // spec_stride
        return start_frame, end_frame + 1

    @staticmethod
    def label_fn(mid_file_name, num_frames, spec_stride):
        """labeling function"""
        label_matrix = np.zeros((num_frames, 88), dtype=np.bool_)
        note_seq = magenta.music.midi_file_to_note_sequence(mid_file_name)
        note_seq = magenta.music.apply_sustain_control_changes(note_seq)
        for note in note_seq.notes:
            assert 21 <= note.pitch <= 108
            note_start_frame, note_end_frame = MiscFns.times_to_frames_fn(
                spec_stride=spec_stride,
                start_time=note.start_time,
                end_time=note.end_time
            )
            label_matrix[note_start_frame:note_end_frame, note.pitch - 21] = True

        return label_matrix

    @staticmethod
    def full_conv_net_kelz(spec_batch, is_training):
        """the Kelz acoustic model [2]"""
        spec_batch.set_shape([None, None, 229])
        outputs = spec_batch[..., None]

        outputs = slim.conv2d(
            inputs=outputs,
            num_outputs=32,
            kernel_size=3,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training),
            scope='conv_0'
        )

        outputs = slim.conv2d(
            inputs=outputs,
            num_outputs=32,
            kernel_size=3,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training),
            scope='conv_1'
        )
        outputs = slim.max_pool2d(inputs=outputs, kernel_size=[1, 2], stride=[1, 2], scope='maxpool_1')
        outputs = slim.dropout(inputs=outputs, keep_prob=0.75, is_training=is_training, scope='dropout_1')

        outputs = slim.conv2d(
            inputs=outputs,
            num_outputs=64,
            kernel_size=3,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training),
            scope='conv_2'
        )
        outputs = slim.max_pool2d(inputs=outputs, kernel_size=[1, 2], stride=[1, 2], scope='maxpool_2')
        outputs = slim.dropout(inputs=outputs, keep_prob=0.75, is_training=is_training, scope='dropout_2')

        dims = tf.shape(outputs)
        outputs = tf.reshape(tensor=outputs, shape=[dims[0], dims[1], outputs.shape[2].value * outputs.shape[3].value],
                             name='flatten_3')
        assert outputs.shape.rank == 3
        outputs = slim.fully_connected(
            inputs=outputs,
            num_outputs=512,
            normalizer_fn=slim.batch_norm,
            normalizer_params=dict(is_training=is_training),
            scope='fc_3'
        )
        assert outputs.shape.as_list() == [None, None, 512]
        outputs = slim.dropout(inputs=outputs, keep_prob=0.5, is_training=is_training, scope='dropout_3')

        outputs = slim.fully_connected(inputs=outputs, num_outputs=88, activation_fn=None, scope='output')

        return outputs

    @staticmethod
    def gen_stats(labels_bool_flattened, logits_bool_flattened, loss):
        """tf code for generating ensemble performance measures and mean loss"""
        assert labels_bool_flattened.dtype == tf.bool
        assert logits_bool_flattened.dtype == tf.bool

        stats_name_to_fn_dict = dict(
            tp=tf.metrics.true_positives,
            fn=tf.metrics.false_negatives,
            tn=tf.metrics.true_negatives,
            fp=tf.metrics.false_positives
        )
        kwargs = dict(
            labels=labels_bool_flattened,
            predictions=logits_bool_flattened
        )
        update_op_list = []
        value_dict = {}
        with tf.name_scope('statistics'):
            for stat_name, stat_fn in stats_name_to_fn_dict.iteritems():
                value_op, update_op = stat_fn(name=stat_name, **kwargs)
                update_op_list.append(update_op)
                value_dict[stat_name] = value_op

            mean_loss_value_op, mean_loss_update_op = tf.metrics.mean(loss, name='average_loss')
            value_dict['average_loss'] = mean_loss_value_op
            update_op_list.append(mean_loss_update_op)

            merged_update_op = tf.group(update_op_list, name='merged_stat_update_op')

        stats_dict = dict(merged_update_op=merged_update_op)
        stats_dict['meta_data_dict'] = {}
        for meta_data_name in ('tp', 'fn', 'tn', 'fp'):
            stats_dict['meta_data_dict'][meta_data_name] = value_dict[meta_data_name]

        stats_dict['average_loss'] = value_dict['average_loss']

        tp = stats_dict['meta_data_dict']['tp']
        fn = stats_dict['meta_data_dict']['fn']
        fp = stats_dict['meta_data_dict']['fp']
        precision = stats_dict['precision'] = tp / (tp + fp + 1e-7)
        recall = stats_dict['recall'] = tp / (tp + fn + 1e-7)
        stats_dict['f1'] = 2. * precision * recall / (precision + recall + 1e-7)

        return stats_dict

    @staticmethod
    def split_train_valid_and_test_files_fn():
        """
        generate non-overlapped training-test set partition

        After downloading and unzipping the MAPS dataset,
        1. define an environment variable called maps to point to the directory of the MAPS dataset,
        2. populate test_dirs with the actual directories of the close and the ambient setting generated by
           the Disklavier piano,
        3. and populate train_dirs with the actual directoreis of the other 7 settings generated by the synthesizer.
        """
        test_dirs = ['ENSTDkCl_2/MUS', 'ENSTDkAm_2/MUS']
        train_dirs = ['AkPnBcht_2/MUS', 'AkPnBsdf_2/MUS', 'AkPnCGdD_2/MUS', 'AkPnStgb_2/MUS',
                      'SptkBGAm_2/MUS', 'SptkBGCl_2/MUS', 'StbgTGd2_2/MUS']
        maps_dir = os.environ['maps']

        test_files = []
        for directory in test_dirs:
            path = os.path.join(maps_dir, directory)
            path = os.path.join(path, '*.wav')
            wav_files = glob.glob(path)
            test_files += wav_files

        test_ids = set([MiscFns.filename_to_id(wav_file) for wav_file in test_files])
        assert len(test_ids) == 53

        training_files = []
        validation_files = []
        for directory in train_dirs:
            path = os.path.join(maps_dir, directory)
            path = os.path.join(path, '*.wav')
            wav_files = glob.glob(path)
            for wav_file in wav_files:
                me_id = MiscFns.filename_to_id(wav_file)
                if me_id not in test_ids:
                    training_files.append(wav_file)
                else:
                    validation_files.append(wav_file)

        assert len(training_files) == 139 and len(test_files) == 60 and len(validation_files) == 71

        return dict(training=training_files, test=test_files, validation=validation_files)

    @staticmethod
    def display_stat_dict_fn(stat_dict):
        """display statistics"""
        for stat_name, stat_value in stat_dict.iteritems():
            if 'individual' in stat_name:
                logging.info(stat_name)
                for idx, sub_value in enumerate(stat_value):
                    logging.info('{} - {}'.format(idx, sub_value))
            else:
                logging.info(stat_name)
                logging.info(stat_value)


# contain all configurations
class Config(object):

    def __init__(self):
        self.debug_mode = DEBUG
        self.test_with_30_secs = False
        self.gpu_id = GPU_ID

        self.num_epochs = 9
        self.batches_per_epoch = 5000
        self.batch_size = 12
        self.learning_rate = 1e-4

        self.train_or_inference = Namespace(
            inference='d0_epoch_9_of_15',
            from_saved=None,
            model_prefix=None
        )
        # inference: point to the saved model for inference
        # from_saved: point to the saved model from which the training continues
        # model_prefix: the prefix used when saving the model
        # order: If inference is not None, then do inference; elif from_saved is not None, then continue training
        #        from the saved model; elif train from scratch.
        #        If model_prefix is None, the model will not be saved.

        self.tb_dir = 'tb_inf'
        # the directory for saving tensorboard data including performance measures, model parameters, and the model itself

        # check if tb_dir exists
        assert self.tb_dir is not None
        tmp_dirs = glob.glob('./*/')
        tmp_dirs = [s[2:-1] for s in tmp_dirs]
        if self.tb_dir in tmp_dirs:
            raise EnvironmentError('\n'
                                   'directory {} for storing tensorboard data already exists!\n'
                                   'Cannot proceed.\n'
                                   'Please specify a different directory.'.format(self.tb_dir)
                                   )

        # check if model exists
        if self.train_or_inference.inference is None and self.train_or_inference.model_prefix is not None:
            if os.path.isdir('./saved_model'):
                tmp_prefixes = glob.glob('./saved_model/*')
                prog = re.compile(r'./saved_model/(.+?)_')
                tmp = []
                for file_name in tmp_prefixes:
                    try:
                        prefix = prog.match(file_name).group(1)
                    except AttributeError:
                        pass
                    else:
                        tmp.append(prefix)
                tmp_prefixes = set(tmp)
                if self.train_or_inference.model_prefix in tmp_prefixes:
                    raise EnvironmentError('\n'
                                           'models with prefix {} already exists.\n'
                                           'Please specify a different prefix.'.format(self.train_or_inference.model_prefix)
                                           )

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.gpu_config = gpu_config

        self.file_names = MiscFns.split_train_valid_and_test_files_fn()

        # in debug mode the numbers of recordings for training, test and validation are minimized for a debugging purpose
        if self.debug_mode:
            # for name in ('training', 'validation', 'test'):
            #     if name == 'training':
            #         del self.file_names[name][2:]
            #     else:
            #         del self.file_names[name][1:]
            self.file_names['training'] = self.file_names['training'][:2]
            self.file_names['validation'] = self.file_names['validation'][:1]
            self.file_names['test'] = self.file_names['test'][0:2]

            self.num_epochs = 3
            self.batches_per_epoch = 5
            self.gpu_id = 0

        # in inference mode, the numbers of recordings for training and validation are minimized
        if self.train_or_inference.inference is not None:
            for name in ('training', 'validation'):
                del self.file_names[name][1:]

        # the logarithmic filterbank
        self.log_filter_bank = MiscFns.log_filter_bank_fn()


# define nn models
class Model(object):
    def __init__(self, config, name):
        assert name in ('validation', 'training', 'test')
        self.name = name
        logging.debug('{} - model - initialize'.format(self.name))
        self.is_training = True if self.name == 'training' else False
        self.config = config

        if not self.is_training:
            self.reinitializable_iter_for_dataset = None
        self.batch = self._gen_batch_fn()  # generate mini-batch

        with tf.name_scope(self.name):
            with tf.variable_scope('full_conv', reuse=tf.AUTO_REUSE):
                logits_stereo = self._nn_model_fn()

            logits_stereo_flattened = flatten_maybe_padded_sequences(
                maybe_padded_sequences=logits_stereo,
                lengths=tf.tile(input=self.batch['num_frames'], multiples=[2]))
            logits_left_flattened, logits_right_flattened = tf.split(
                value=logits_stereo_flattened, num_or_size_splits=2, axis=0)
            logits_minor_flattened = tf.minimum(logits_left_flattened, logits_right_flattened)
            logits_larger_flattened = tf.maximum(logits_left_flattened, logits_right_flattened)
            labels_bool_flattened = flatten_maybe_padded_sequences(
                maybe_padded_sequences=self.batch['label'], lengths=self.batch['num_frames'])
            negated_labels_bool_flattened = tf.logical_not(labels_bool_flattened)
            labels_float_flattened = tf.cast(x=labels_bool_flattened, dtype=tf.float32)
            logits_mono_flattened = tf.where(
                tf.equal(labels_bool_flattened, True), logits_minor_flattened, logits_larger_flattened)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_float_flattened,
                                                           logits=logits_mono_flattened)
            loss = tf.reduce_mean(loss)

            if self.is_training:
                _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if _update_ops:
                    with tf.control_dependencies(_update_ops):
                        training_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)
                else:
                    training_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)

            pred_labels_flattened = tf.greater(logits_left_flattened + logits_right_flattened, 0.)
            negated_pred_labels_flattened = tf.logical_not(pred_labels_flattened)

            # individual and ensemble statistics for test and validation
            if not self.is_training:
                with tf.name_scope('individual_and_ensemble_stats'):
                    with tf.variable_scope(
                            '{}_local_vars'.format(self.name), reuse=tf.AUTO_REUSE):
                        individual_tps_fps_tns_fns_var = tf.get_variable(
                            name='individual_tps_fps_tns_fns',
                            shape=[len(self.config.file_names[self.name]), 4],
                            dtype=tf.int32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )

                        acc_loss_var = tf.get_variable(
                            name='acc_loss',
                            shape=[],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )

                        batch_counter_var = tf.get_variable(
                            name='batch_counter',
                            shape=[],
                            dtype=tf.int32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )

                    loop_var_proto = collections.namedtuple(
                        'loop_var_proto',
                        ['sample_idx', 'batch_size', 'preds', 'negated_preds',
                         'labels', 'negated_labels', 'lengths', 'me_ids'])

                    def cond_fn(loop_var):
                        return tf.less(loop_var.sample_idx, loop_var.batch_size)

                    def body_fn(loop_var):
                        start_pos = tf.reduce_sum(loop_var.lengths[:loop_var.sample_idx])
                        end_pos = start_pos + loop_var.lengths[loop_var.sample_idx]
                        cur_preds = loop_var.preds
                        negated_cur_preds = loop_var.negated_preds
                        cur_labels = loop_var.labels
                        negated_cur_labels = loop_var.negated_labels
                        cur_preds, negated_cur_preds, cur_labels, negated_cur_labels = \
                            [value[start_pos:end_pos]
                             for value in [cur_preds, negated_cur_preds, cur_labels, negated_cur_labels]]
                        tps = tf.logical_and(cur_preds, cur_labels)
                        fps = tf.logical_and(cur_preds, negated_cur_labels)
                        tns = tf.logical_and(negated_cur_preds, negated_cur_labels)
                        fns = tf.logical_and(negated_cur_preds, cur_labels)
                        tps, fps, tns, fns = \
                            [tf.reduce_sum(tf.cast(value, tf.int32)) for value in [tps, fps, tns, fns]]
                        me_id = loop_var.me_ids[loop_var.sample_idx]
                        stats_var = individual_tps_fps_tns_fns_var
                        _new_value = stats_var[me_id] + tf.convert_to_tensor([tps, fps, tns, fns])
                        _update_stats = tf.scatter_update(
                            stats_var, me_id, _new_value, use_locking=True)
                        with tf.control_dependencies([_update_stats]):
                            sample_idx = loop_var.sample_idx + 1
                        loop_var = loop_var_proto(
                            sample_idx=sample_idx,
                            batch_size=loop_var.batch_size,
                            preds=loop_var.preds,
                            negated_preds=loop_var.negated_preds,
                            labels=loop_var.labels,
                            negated_labels=loop_var.negated_labels,
                            lengths=loop_var.lengths,
                            me_ids=loop_var.me_ids
                        )

                        return [loop_var]

                    sample_idx = tf.constant(0, dtype=tf.int32)
                    cur_batch_size = tf.shape(self.batch['num_frames'])[0]
                    loop_var = loop_var_proto(
                        sample_idx=sample_idx,
                        batch_size=cur_batch_size,
                        preds=pred_labels_flattened,
                        negated_preds=negated_pred_labels_flattened,
                        labels=labels_bool_flattened,
                        negated_labels=negated_labels_bool_flattened,
                        lengths=self.batch['num_frames'],
                        me_ids=self.batch['me_id']
                    )
                    final_sample_idx = tf.while_loop(
                        cond=cond_fn,
                        body=body_fn,
                        loop_vars=[loop_var],
                        parallel_iterations=self.config.batch_size,
                        back_prop=False,
                        return_same_structure=True
                    )[0].sample_idx

                    individual_tps_fps_tns_fns_float = tf.cast(individual_tps_fps_tns_fns_var, tf.float32)
                    tps, fps, _, fns = tf.unstack(individual_tps_fps_tns_fns_float, axis=1)
                    me_wise_precisions = tps / (tps + fps + 1e-7)
                    me_wise_recalls = tps / (tps + fns + 1e-7)
                    me_wise_f1s = 2. * me_wise_precisions * me_wise_recalls / \
                                  (me_wise_precisions + me_wise_recalls + 1e-7)
                    me_wise_prfs = tf.stack([me_wise_precisions, me_wise_recalls, me_wise_f1s], axis=1)
                    assert me_wise_prfs.shape.as_list() == [len(self.config.file_names[self.name]), 3]
                    average_me_wise_prf = tf.reduce_mean(me_wise_prfs, axis=0)
                    assert average_me_wise_prf.shape.as_list() == [3]

                    # ensemble stats
                    ensemble_tps_fps_tns_fns = tf.reduce_sum(individual_tps_fps_tns_fns_var, axis=0)
                    tps, fps, _, fns = tf.unstack(tf.cast(ensemble_tps_fps_tns_fns, tf.float32))
                    en_precision = tps / (tps + fps + 1e-7)
                    en_recall = tps / (tps + fns + 1e-7)
                    en_f1 = 2. * en_precision * en_recall / (en_precision + en_recall + 1e-7)
                    batch_counter_update_op = tf.assign_add(batch_counter_var, 1)
                    acc_loss_update_op = tf.assign_add(acc_loss_var, loss)
                    ensemble_prf_and_loss = tf.convert_to_tensor(
                        [en_precision, en_recall, en_f1, acc_loss_var / tf.cast(batch_counter_var, tf.float32)])

                    update_op_after_each_batch = tf.group(
                        final_sample_idx, batch_counter_update_op, acc_loss_update_op,
                        name='grouped update ops to be run after each batch'.replace(' ', '_'))
                    stats_after_each_epoch = dict(
                        individual_tps_fps_tns_fns=individual_tps_fps_tns_fns_var,
                        individual_prfs=me_wise_prfs,
                        ensemble_tps_fps_tns_fns=ensemble_tps_fps_tns_fns,
                        ensemble_prf_and_loss=ensemble_prf_and_loss,
                        average_prf=average_me_wise_prf
                    )

            # ensemble stats for training
            if self.is_training:
                with tf.name_scope('ensemble_stats'):
                    with tf.variable_scope(
                            '{}_local_vars'.format(self.name), reuse=tf.AUTO_REUSE):
                        ensemble_tps_fps_tns_fns_var = tf.get_variable(
                            name='ensemble_tps_fps_tns_fns',
                            shape=[4],
                            dtype=tf.int32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )
                        acc_loss_var = tf.get_variable(
                            name='acc_loss',
                            shape=[],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )
                        batch_counter_var = tf.get_variable(
                            name='batch_counter',
                            shape=[],
                            dtype=tf.int32,
                            initializer=tf.zeros_initializer,
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES]
                        )

                    tps = tf.logical_and(pred_labels_flattened, labels_bool_flattened)
                    fps = tf.logical_and(pred_labels_flattened, negated_labels_bool_flattened)
                    tns = tf.logical_and(negated_pred_labels_flattened, negated_labels_bool_flattened)
                    fns = tf.logical_and(negated_pred_labels_flattened, labels_bool_flattened)
                    tps, fps, tns, fns = [tf.reduce_sum(tf.cast(value, tf.int32)) for value in [tps, fps, tns, fns]]
                    ensemble_tps_fps_tns_fns_update_op = tf.assign_add(
                        ensemble_tps_fps_tns_fns_var, tf.convert_to_tensor([tps, fps, tns, fns]))
                    acc_loss_update_op = tf.assign_add(acc_loss_var, loss)
                    batch_counter_update_op = tf.assign_add(batch_counter_var, 1)
                    ensemble_tps_fps_tns_fns_float = tf.cast(ensemble_tps_fps_tns_fns_var, tf.float32)
                    tps, fps, _, fns = tf.unstack(ensemble_tps_fps_tns_fns_float)
                    ensemble_precision = tps / (tps + fps + 1e-7)
                    ensemble_recall = tps / (tps + fns + 1e-7)
                    ensemble_f1 = 2. * ensemble_precision * ensemble_recall / \
                                  (ensemble_precision + ensemble_recall + 1e-7)
                    ensemble_loss = acc_loss_var / tf.cast(batch_counter_var, tf.float32)
                    ensemble_prf_and_loss = tf.convert_to_tensor(
                        [ensemble_precision, ensemble_recall, ensemble_f1, ensemble_loss])

                    update_op_after_each_batch = tf.group(
                        batch_counter_update_op, ensemble_tps_fps_tns_fns_update_op, acc_loss_update_op)
                    stats_after_each_epoch = dict(
                        ensemble_tps_fps_tns_fns=ensemble_tps_fps_tns_fns_var,
                        ensemble_prf_and_loss=ensemble_prf_and_loss
                    )

            # define tensorboard summaries
            with tf.name_scope('tensorboard_summary'):
                with tf.name_scope('statistics'):
                    if not self.is_training:
                        list_of_summaries = []
                        with tf.name_scope('ensemble'):
                            p, r, f, lo = tf.unstack(stats_after_each_epoch['ensemble_prf_and_loss'])
                            items_for_summary = dict(precision=p, recall=r, f1=f, average_loss=lo)
                            for item_name, item_value in items_for_summary.iteritems():
                                tmp = tf.summary.scalar(item_name, item_value)
                                list_of_summaries.append(tmp)
                        with tf.name_scope('individual'):
                            p, r, f = tf.unstack(stats_after_each_epoch['average_prf'])
                            items_for_summary = dict(precision=p, recall=r, f1=f)
                            for item_name, item_value in items_for_summary.iteritems():
                                tmp = tf.summary.scalar(item_name, item_value)
                                list_of_summaries.append(tmp)
                    else:
                        list_of_summaries = []
                        with tf.name_scope('ensemble'):
                            p, r, f, lo = tf.unstack(stats_after_each_epoch['ensemble_prf_and_loss'])
                            items_for_summary = dict(precision=p, recall=r, f1=f, average_loss=lo)
                            for item_name, item_value in items_for_summary.iteritems():
                                tmp = tf.summary.scalar(item_name, item_value)
                                list_of_summaries.append(tmp)

                    statistical_summary = tf.summary.merge(list_of_summaries)

                with tf.name_scope('images'):
                    image_summary_length = int(6 * 16000 // 512)
                    labels_uint8 = self.batch['label'][:, :image_summary_length, :]
                    labels_uint8 = tf.cast(labels_uint8, tf.uint8) * 255
                    assert labels_uint8.dtype == tf.uint8
                    labels_uint8 = labels_uint8[..., None]

                    _logits_left = tf.split(value=logits_stereo, num_or_size_splits=2, axis=0)[0]
                    logits_prob_uint8 = tf.sigmoid(_logits_left[:, :image_summary_length, :])
                    logits_prob_uint8 = tf.cast(logits_prob_uint8 * 255., tf.uint8)
                    logits_prob_uint8 = logits_prob_uint8[..., None]

                    images = tf.concat([labels_uint8, logits_prob_uint8, tf.zeros_like(labels_uint8)], axis=-1)
                    images = tf.transpose(images, [0, 2, 1, 3])
                    images.set_shape([None, 88, image_summary_length, 3])
                    image_summary = tf.summary.image('images', images)

                if self.is_training:
                    with tf.name_scope('params'):
                        var_summary_dict = dict()
                        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                            var_summary_dict[var.op.name] = tf.summary.histogram(var.op.name, var)
                        param_summary = tf.summary.merge(var_summary_dict.values())

        if self.is_training:
            op_dict = dict(
                training_op=training_op,
                tb_summary=dict(statistics=statistical_summary, image=image_summary, parameter=param_summary),
                update_op_after_each_batch=update_op_after_each_batch,
                statistics_after_each_epoch=stats_after_each_epoch
            )
        else:
            op_dict = dict(
                tb_summary=dict(statistics=statistical_summary, image=image_summary),
                update_op_after_each_batch=update_op_after_each_batch,
                statistics_after_each_epoch=stats_after_each_epoch
            )

        self.op_dict = op_dict

    def _dataset_iter_fn(self):
        """dataset generator"""

        logging.debug('{} - enter generator'.format(self.name))
        if not hasattr(self, 'dataset'):
            file_names = self.config.file_names[self.name]
            if self.name == 'test' and self.config.test_with_30_secs:
                _duration = 30
            else:
                _duration = None
            logging.debug('{} - generate spectrograms and labels'.format(self.name))
            dataset = []
            for file_idx, wav_file_name in enumerate(file_names):
                logging.debug('{}/{} - {}'.format(
                    file_idx + 1, len(file_names),
                    os.path.basename(wav_file_name))
                )
                samples, unused_sr = librosa.load(
                    mono=False, path=wav_file_name, sr=16000, duration=_duration, dtype=np.float32)
                assert unused_sr == 16000
                assert samples.shape[0] == 2
                spectrogram = []
                for ch in xrange(2):
                    sg = MiscFns.spectrogram_fn(
                        samples=samples[ch],
                        log_filter_bank_basis=self.config.log_filter_bank,
                        spec_stride=512
                    )
                    spectrogram.append(sg)
                spectrogram = np.stack(spectrogram, axis=-1)
                assert spectrogram.shape[1:] == (229, 2)
                mid_file_name = wav_file_name[:-4] + '.mid'
                label = MiscFns.label_fn(mid_file_name=mid_file_name, num_frames=spectrogram.shape[0], spec_stride=512)
                dataset.append([spectrogram, label])

                logging.debug('number of frames - {}'.format(spectrogram.shape[0]))
            self.dataset = dataset

            rec_start_end_for_shuffle = []
            for rec_idx, rec_dict in enumerate(self.dataset):
                num_frames = len(rec_dict[0])
                split_frames = range(0, num_frames + 1, 900)
                if split_frames[-1] != num_frames:
                    split_frames.append(num_frames)
                start_end_frame_pairs = zip(split_frames[:-1], split_frames[1:])
                rec_start_end_idx_list = [[rec_idx] + list(start_end_pair) for start_end_pair in start_end_frame_pairs]
                rec_start_end_for_shuffle += rec_start_end_idx_list
            self.rec_start_end_for_shuffle = rec_start_end_for_shuffle
        else:
            logging.debug('{} - generator - dataset already exists'.format(self.name))

        logging.debug('{} - generator begins'.format(self.name))
        if self.is_training:
            np.random.shuffle(self.rec_start_end_for_shuffle)
        for rec_idx, start_frame, end_frame in self.rec_start_end_for_shuffle:
            rec_dict = self.dataset[rec_idx]
            yield dict(
                spectrogram=rec_dict[0][start_frame:end_frame],
                label=rec_dict[1][start_frame:end_frame],
                num_frames=end_frame - start_frame,
                me_id=rec_idx
            )
        logging.debug('{} - generator ended'.format(self.name))

    def _gen_batch_fn(self):
        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_generator(
                generator=self._dataset_iter_fn,
                output_types=dict(spectrogram=tf.float32, label=tf.bool, num_frames=tf.int32, me_id=tf.int32),
                output_shapes=dict(
                    spectrogram=[None, 229, 2],
                    label=[None, 88],
                    num_frames=[],
                    me_id=[]
                )
            )

            if self.is_training:
                dataset = dataset.repeat()

            dataset = dataset.padded_batch(
                batch_size=self.config.batch_size,
                padded_shapes=dict(
                    spectrogram=[-1, 229, 2],
                    label=[-1, 88],
                    num_frames=[],
                    me_id=[]
                )
            )

            dataset = dataset.prefetch(5)

            if self.is_training:
                dataset_iter = dataset.make_one_shot_iterator()
                element = dataset_iter.get_next()
            else:
                reinitializabel_iter = dataset.make_initializable_iterator()
                self.reinitializable_iter_for_dataset = reinitializabel_iter
                element = reinitializabel_iter.get_next()

        return element

    def _nn_model_fn(self):
        inputs = self.batch['spectrogram']
        assert inputs.shape.as_list() == [None, None, 229, 2]

        # treat the two sound channels as independent examples
        inputs = tf.concat(tf.split(value=inputs, num_or_size_splits=2, axis=-1), axis=0)
        inputs = tf.squeeze(inputs, axis=-1)
        assert inputs.shape.as_list() == [None, None, 229]
        outputs = MiscFns.full_conv_net_kelz(
            spec_batch=inputs,
            is_training=self.is_training
        )

        return outputs


def main():
    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()  # generate configurations

    # generate models
    for name in ('training', 'validation', 'test'):
        MODEL_DICT[name] = Model(config=MODEL_DICT['config'], name=name)

    # placeholder for auxiliary information
    aug_info_pl = tf.placeholder(dtype=tf.string, name='aug_info_pl')
    aug_info_summary = tf.summary.text('aug_info_summary', aug_info_pl)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(MODEL_DICT['config'].gpu_id)
    with tf.Session(config=MODEL_DICT['config'].gpu_config) as sess:

        # define model saver
        if MODEL_DICT['config'].train_or_inference.inference is not None or \
                MODEL_DICT['config'].train_or_inference.from_saved is not None or \
                MODEL_DICT['config'].train_or_inference.model_prefix is not None:
            MODEL_DICT['model_saver'] = tf.train.Saver(max_to_keep=200)

            logging.info('saved/restored variables:')
            for idx, var in enumerate(MODEL_DICT['model_saver']._var_list):
                logging.info('{}\t{}'.format(idx, var.op.name))

        # define summary writers
        summary_writer_dict = {}
        for training_valid_or_test in ('training', 'validation', 'test'):
            if training_valid_or_test == 'training':
                summary_writer_dict[training_valid_or_test] = tf.summary.FileWriter(
                    os.path.join(MODEL_DICT['config'].tb_dir, training_valid_or_test),
                    sess.graph
                )
            else:
                summary_writer_dict[training_valid_or_test] = tf.summary.FileWriter(
                    os.path.join(MODEL_DICT['config'].tb_dir, training_valid_or_test)
                )

        aug_info = []
        if MODEL_DICT['config'].train_or_inference.inference is not None:
            aug_info.append('inference with {}'.format(MODEL_DICT['config'].train_or_inference.inference))
            aug_info.append('inference with only the first 30 secs - {}'.format(MODEL_DICT['config'].test_with_30_secs))
        elif MODEL_DICT['config'].train_or_inference.from_saved is not None:
            aug_info.append('continue training from {}'.format(MODEL_DICT['config'].train_or_inference.from_saved))
        aug_info.append('learning rate - {}'.format(MODEL_DICT['config'].learning_rate))
        aug_info.append('tb dir - {}'.format(MODEL_DICT['config'].tb_dir))
        aug_info.append('debug mode - {}'.format(MODEL_DICT['config'].debug_mode))
        aug_info.append('batch size - {}'.format(MODEL_DICT['config'].batch_size))
        aug_info.append('num of batches per epoch - {}'.format(MODEL_DICT['config'].batches_per_epoch))
        aug_info.append('num of epochs - {}'.format(MODEL_DICT['config'].num_epochs))
        aug_info.append('training start time - {}'.format(datetime.datetime.now()))
        aug_info = '\n\n'.join(aug_info)
        logging.info(aug_info)
        summary_writer_dict['training'].add_summary(sess.run(aug_info_summary, feed_dict={aug_info_pl: aug_info}))

        logging.info('global vars -')
        for idx, var in enumerate(tf.global_variables()):
            logging.info("{}\t{}\t{}".format(idx, var.name, var.shape))

        logging.info('local vars -')
        for idx, var in enumerate(tf.local_variables()):
            logging.info('{}\t{}'.format(idx, var.name))

        # extract tf operations
        op_stat_summary_dict = {}
        for training_valid_or_test in ('training', 'validation', 'test'):
            op_list = []
            if training_valid_or_test == 'training':
                op_list.append(MODEL_DICT[training_valid_or_test].op_dict['training_op'])
                op_list.append(MODEL_DICT[training_valid_or_test].op_dict['update_op_after_each_batch'])
            else:
                op_list.append(MODEL_DICT[training_valid_or_test].op_dict['update_op_after_each_batch'])

            stat_op_dict = MODEL_DICT[training_valid_or_test].op_dict['statistics_after_each_epoch']

            tb_summary_dict = MODEL_DICT[training_valid_or_test].op_dict['tb_summary']

            op_stat_summary_dict[training_valid_or_test] = dict(
                op_list=op_list,
                stat_op_dict=stat_op_dict,
                tb_summary_dict=tb_summary_dict
            )

        if MODEL_DICT['config'].train_or_inference.inference is not None:  # inference
            save_path = os.path.join('saved_model', MODEL_DICT['config'].train_or_inference.inference)
            MODEL_DICT['model_saver'].restore(sess, save_path)

            logging.info('do inference ...')
            # initialize local variables for storing statistics
            sess.run(tf.initializers.variables(tf.local_variables()))
            # initialize dataset iterator
            sess.run(MODEL_DICT['test'].reinitializable_iter_for_dataset.initializer)

            op_list = op_stat_summary_dict['test']['op_list']
            stat_op_dict = op_stat_summary_dict['test']['stat_op_dict']
            tb_summary_image = op_stat_summary_dict['test']['tb_summary_dict']['image']
            tb_summary_stats = op_stat_summary_dict['test']['tb_summary_dict']['statistics']

            batch_idx = 0
            op_list_with_image_summary = [tb_summary_image] + op_list
            logging.info('batch - {}'.format(batch_idx + 1))
            tmp = sess.run(op_list_with_image_summary)
            images = tmp[0]
            summary_writer_dict['test'].add_summary(images, 0)

            while True:
                try:
                    sess.run(op_list)
                except tf.errors.OutOfRangeError:
                    break
                else:
                    batch_idx += 1
                    logging.info('batch - {}'.format(batch_idx + 1))
            # write summary data
            summary_writer_dict[training_valid_or_test].add_summary(sess.run(tb_summary_stats), 0)

            # generate statistics
            stat_dict = sess.run(stat_op_dict)

            # display statistics
            MiscFns.display_stat_dict_fn(stat_dict)
        elif MODEL_DICT['config'].train_or_inference.from_saved is not None:  # restore saved model for training
            save_path = os.path.join('saved_model', MODEL_DICT['config'].train_or_inference.from_saved)
            MODEL_DICT['model_saver'].restore(sess, save_path)

            # reproduce statistics
            logging.info('reproduce results ...')
            sess.run(tf.initializers.variables(tf.local_variables()))
            for valid_or_test in ('validation', 'test'):
                sess.run(MODEL_DICT[valid_or_test].reinitializable_iter_for_dataset.initializer)
            for valid_or_test in ('validation', 'test'):
                logging.info(valid_or_test)

                op_list = op_stat_summary_dict[valid_or_test]['op_list']
                stat_op_dict = op_stat_summary_dict[valid_or_test]['stat_op_dict']
                statistical_summary = op_stat_summary_dict[valid_or_test]['tb_summary_dict']['statistics']
                image_summary = op_stat_summary_dict[valid_or_test]['tb_summary_dict']['image']

                batch_idx = 0
                op_list_with_image_summary = [image_summary] + op_list
                logging.info('batch - {}'.format(batch_idx + 1))
                tmp = sess.run(op_list_with_image_summary)
                images = tmp[0]
                summary_writer_dict[valid_or_test].add_summary(images, 0)

                while True:
                    try:
                        sess.run(op_list)
                    except tf.errors.OutOfRangeError:
                        break
                    else:
                        batch_idx += 1
                        logging.info('batch - {}'.format(batch_idx + 1))

                summary_writer_dict[valid_or_test].add_summary(sess.run(statistical_summary), 0)

                stat_dict = sess.run(stat_op_dict)

                MiscFns.display_stat_dict_fn(stat_dict)
        else:  # train from scratch and need to initialize global variables
            sess.run(tf.initializers.variables(tf.global_variables()))

        if MODEL_DICT['config'].train_or_inference.inference is None:
            for training_valid_test_epoch_idx in xrange(MODEL_DICT['config'].num_epochs):
                logging.info('\n\nepoch - {}/{}'.format(training_valid_test_epoch_idx + 1, MODEL_DICT['config'].num_epochs))

                sess.run(tf.initializers.variables(tf.local_variables()))

                # to enable prefetch
                for valid_or_test in ('validation', 'test'):
                    sess.run(MODEL_DICT[valid_or_test].reinitializable_iter_for_dataset.initializer)

                for training_valid_or_test in ('training', 'validation', 'test'):
                    logging.info(training_valid_or_test)

                    op_list = op_stat_summary_dict[training_valid_or_test]['op_list']
                    stat_op_dict = op_stat_summary_dict[training_valid_or_test]['stat_op_dict']
                    statistical_summary = op_stat_summary_dict[training_valid_or_test]['tb_summary_dict']['statistics']
                    image_summary = op_stat_summary_dict[training_valid_or_test]['tb_summary_dict']['image']

                    if training_valid_or_test == 'training':

                        for batch_idx in xrange(MODEL_DICT['config'].batches_per_epoch):
                            sess.run(op_list)
                            logging.debug('batch - {}/{}'.format(batch_idx + 1, MODEL_DICT['config'].batches_per_epoch))

                        summary_writer_dict[training_valid_or_test].add_summary(
                            sess.run(image_summary), training_valid_test_epoch_idx + 1)
                        summary_writer_dict[training_valid_or_test].add_summary(
                            sess.run(statistical_summary), training_valid_test_epoch_idx + 1)
                        param_summary = MODEL_DICT[training_valid_or_test].op_dict['tb_summary']['parameter']
                        summary_writer_dict[training_valid_or_test].add_summary(
                            sess.run(param_summary), training_valid_test_epoch_idx + 1)

                        stat_dict = sess.run(stat_op_dict)

                        if MODEL_DICT['config'].train_or_inference.model_prefix is not None:
                            save_path = MODEL_DICT['config'].train_or_inference.model_prefix + \
                                        '_' + 'epoch_{}_of_{}'.format(training_valid_test_epoch_idx + 1,
                                                                      MODEL_DICT['config'].num_epochs)
                            save_path = os.path.join('saved_model', save_path)
                            save_path = MODEL_DICT['model_saver'].save(
                                sess=sess,
                                save_path=save_path,
                                global_step=None,
                                write_meta_graph=False
                            )
                            logging.info('model saved to {}'.format(save_path))
                    else:
                        batch_idx = 0
                        op_list_with_image_summary = [image_summary] + op_list
                        logging.debug('batch - {}'.format(batch_idx + 1))
                        tmp = sess.run(op_list_with_image_summary)
                        images = tmp[0]
                        summary_writer_dict[training_valid_or_test].add_summary(
                            images,
                            training_valid_test_epoch_idx + 1
                        )

                        while True:
                            try:
                                sess.run(op_list)
                            except tf.errors.OutOfRangeError:
                                break
                            else:
                                batch_idx += 1
                                logging.debug('batch - {}'.format(batch_idx + 1))

                        summary_writer_dict[training_valid_or_test].add_summary(
                            sess.run(statistical_summary),
                            training_valid_test_epoch_idx + 1
                        )

                        stat_dict = sess.run(stat_op_dict)

                    MiscFns.display_stat_dict_fn(stat_dict)

        msg = 'training end time - {}'.format(datetime.datetime.now())
        logging.info(msg)
        summary_writer_dict['training'].add_summary(sess.run(aug_info_summary, feed_dict={aug_info_pl: msg}))

        for training_valid_or_test in ('training', 'validation', 'test'):
            summary_writer_dict[training_valid_or_test].close()


if __name__ == '__main__':
    main()

















