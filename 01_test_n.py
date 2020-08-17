"""
 @file   01_test.py
 @brief  Script for test
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import csv
import re
import itertools
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
# from import
from tqdm import tqdm
from sklearn import metrics
# original lib
import common_cnn_n as com
import model
import tensorflow as tf
import evaluation

import librosa
from pathlib import Path
import keras
import model
import probability

########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################


########################################################################
# select gpu
########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


#######################################################################

# tf.debugging.set_log_device_placement(True)
#
# gpus = tf.config.experimental.list_logical_devices('GPU')

########################################################################
# def
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    if os.path.split(target_dir)[1] == 'washer':  # if True, it is smi washer data
        machine_id_list = ['LG.motor.washer']
        return machine_id_list

    elif os.path.split(target_dir)[1] == 'ToyCar':
        machine_id_list = ['ToyCar']
        return machine_id_list

    # create test files
    dir_path = os.path.abspath(
        "{dir}/{dir_name}/*/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))

    return machine_id_list


def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav",
                             ):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    com.logger.info("target_dir : {}".format(target_dir + "_" + id_name))
    if os.path.split(target_dir)[1] == 'washer':  # if True, it is smi washer data
        prefix_normal = 'O'
        prefix_anomaly = 'X'
        dir_name = 'train'  # data for train and test are same

    elif os.path.split(target_dir)[1] == 'ToyCar':
        prefix_anomaly = 'ab'

    # development
    if mode:
        # normal files
        normal_files = sorted(
            glob.glob(f"{target_dir}/{dir_name}/normal/*{prefix_normal}*.{ext}"))
        normal_labels = numpy.zeros(len(normal_files))

        # anomaly files
        anomaly_files = sorted(
            glob.glob(f"{target_dir}/{dir_name}/normal/*{prefix_anomaly}*.{ext}"))
        anomaly_labels = numpy.ones(len(anomaly_files))

        noise_files = sorted(
            glob.glob(f'{target_dir}/{dir_name}/noise/*.{ext}'))

        files = numpy.concatenate((normal_files, anomaly_files, noise_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)

        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception(f"no_{ext}_file!!")
        print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*/*{id_name}*.{ext}".format(dir=target_dir,
                                                                    dir_name=dir_name,
                                                                    id_name=id_name,
                                                                    ext=ext)))
        labels = None
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception(f"no_{ext}_file!!")
        print("\n=========================================")

    return files, labels


def make_target(path,
                y_true,
                mode):
    clean_dir = param['clean_directory']
    file_name = path.split('/')[-1]

    if y_true == 0:  # normal
        type = 'normal'
    else:  # anomaly
        type = 'anomaly'

    y, sr = librosa.load(f'{clean_dir}/{type}/{file_name}', sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=param['feature']['n_fft'],
                                                     hop_length=param['feature']['hop_length'],
                                                     n_mels=param['feature']['n_mels'],
                                                     power=param['feature']['power'])
    log_mel_spectrogram = librosa.core.power_to_db(mel_spectrogram)[:, :]

    if mode == 'cnn':
        # B, F, T, C
        target = numpy.expand_dims(log_mel_spectrogram, axis=(0, -1))

    else:
        frames = param['feature']['frames']
        n_mels = param['feature']['n_mels']
        dims = n_mels * frames

        target_size = len(log_mel_spectrogram[0, :]) - frames + 1

        if target_size < 1:
            return numpy.empty((0, dims), float)

        # 06 generate feature vectors by concatenating multi_frames
        target = numpy.zeros((target_size, dims), float)
        for t in range(frames):
            target[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:,
                                                      t: t + target_size].T

    return target


import numpy as np
from keras.models import load_model

import preprocess


def generate_audio_from_model_output(
        path_to_model, input_audio_size, generated_frames,
        ):
    wavenet = load_model(path_to_model)
    # We initialize with zeros, can also use a proper seed.
    generated_audio = np.zeros(input_audio_size, dtype=np.int16)
    cur_frame = 0
    while cur_frame < generated_frames:
        # Frame is always shifting by `cur_frame`, so that we can always
        # get the last `input_audio_size` values.
        probability_distribution = wavenet.predict(
            generated_audio[cur_frame:].reshape(
                1, input_audio_size, 1)).flatten()
        cur_sample = preprocess.prediction_to_waveform_value(probability_distribution)
        generated_audio = np.append(generated_audio, cur_sample)
        cur_frame += 1
    return generated_audio


########################################################################


########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    gpus = tf.config.experimental.list_logical_devices('GPU')

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)
    os.makedirs(param["save_directory"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print("============== MODEL LOAD ==============")
        # set model path
        model_file = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                         machine_type=machine_type)

        # load model file
        # if not os.path.exists(model_file + '_model.hdf5'):
        #     com.logger.error("{} model not found ".format(machine_type))
        #     sys.exit(-1)

        # load detector
        # detector = keras.models.load_model(model_file + '_model.hdf5')
        # detector.summary()

        detector = keras.models.load_model(param['model_directory'] + '/weights.00001000.h5')
        detector.summary()

        data_path = Path(f'{param["data_directory"]}')
        mean = numpy.load(data_path / 'org_data_mean.npy')  # (B, D) or (B, F, T, C)
        std = numpy.load(data_path / 'org_data_std.npy')
        mu = numpy.load(data_path / 'mu.npy')
        sigma = numpy.load(data_path / 'sigma.npy')

        machine_id_list = get_machine_id_list_for_test(target_dir)
        denoised_dict = {}
        reconst_dict = {}
        for id_str in machine_id_list:
            # load test file
            test_files, y_true = test_file_list_generator(target_dir, id_name=id_str, ext='wav')

            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
            if os.path.exists(Path(param['save_directory'])/'result.npz'):  # if data exists
                com.logger.info("test result exists")
                result = numpy.load(Path(param['save_directory']) / 'result.npz', allow_pickle=True)
                denoising_score = result['denoising_score']
                reconst_score = result['reconst_score']
                noise_list = result['noise_list']
                denoised_dict = result['denoised_dict'].item()
                reconst_dict = result['reconst_dict'].item()

            else:
                denoising_score = [0. for k in test_files]
                reconst_score = [0. for k in test_files]
                noise_list = [0. for k in test_files]
                for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
                    try:
                        _, data = com.file_to_vector_array(file_path,  # B, dims or C, F, T
                                                           noise_file_name=None,
                                                           **param["feature"])

                        # reshape data if data shape is C, F, T -> 1, F, T, C
                        # data = numpy.reshape(data, (1, data.shape[1], data.shape[2], data.shape[0]))
                        # target = make_target(file_path, y_true[file_idx], mode='cnn')
                        #
                        # noise = numpy.mean(numpy.square(data - target))
                        # noise_list[file_idx] = noise

                        for i in range(data.shape[1]):
                            data[:, i, :] = (data[:, i, :] - mean[i]) / std[i]

                        if param['denoising'] == 'none':
                            data = numpy.moveaxis(data, 1, 2)
                            reconst = detector.predict(data, batch_size=data.shape[0])
                            reconst_dict[file_path] = reconst
                            # reconst_score[file_idx] = numpy.mean(numpy.square(data[:, 31:, :] - reconst))
                            x = probability.calc_x(data[:, 31:, :], reconst)
                            prob = probability.get_prob(x, mu, sigma)
                            reconst_score[file_idx] = prob

                        else:
                            # denoised = denoiser.predict(data, batch_size=data.shape[0])
                            # denoised_dict[file_path] = denoised
                            # denoising_score[file_idx] = numpy.mean(numpy.square(target - denoised))

                            # reconst = detector.predict(denoised, batch_size=data.shape[0])
                            reconst_dict[file_path] = reconst
                            reconst_score[file_idx] = numpy.mean(numpy.square(data - reconst))

                    except Exception as e:
                        print(e)
                        com.logger.error("file broken!!: {}".format(file_path))

                numpy.savez(Path(param['save_directory']) / 'result.npz',
                            denoising_score=denoising_score,
                            reconst_score=reconst_score,
                            noise_list=noise_list,
                            denoised_dict=denoised_dict,
                            reconst_dict=reconst_dict)

            if mode:
                auc = metrics.roc_auc_score(y_true, reconst_score)
                p_auc = metrics.roc_auc_score(y_true, reconst_score, max_fpr=param["max_fpr"])
                com.logger.info("AUC : {}".format(auc))
                com.logger.info("pAUC : {}".format(p_auc))

            print("\n============ END OF TEST FOR A MACHINE ID ============")

        if mode:
            writer = tf.summary.create_file_writer(param['log_dir'])

            with writer.as_default():
                # add to tensorboard
                tf.summary.image('roc curve', evaluation.draw_roc_curve(y_true, reconst_score), step=0)
                tf.summary.image('Precision-Recall curve',
                                 evaluation.draw_PRcurve(y_true, reconst_score), step=0)
                tf.summary.image('reconstruction error histogram',
                                 evaluation.plot_hist(y_true, reconst_score, log=False), step=0)
                tf.summary.image('denoise and noise histogram',
                                 evaluation.plot_multi_hist(y_true, noise_list, denoising_score,
                                                            log=False), step=0)

                if param['denoising'] != 'none':

                    tf.summary.image('denoise score histogram',
                                     evaluation.plot_hist(y_true, denoising_score, log=False), step=0)

                    file_path_dict = evaluation.select_data(denoising_score,
                                                            denoised_dict)

                    for type, path_list in file_path_dict.items():
                        spec_fig_list = evaluation.\
                            draw_true_spectrogram(path_list,
                                                  n_mels=param["feature"]["n_mels"],
                                                  n_fft=param["feature"]["n_fft"],
                                                  hop_length=param["feature"]["hop_length"],
                                                  )
                        pred_spec_fig_list = evaluation.\
                            draw_predict_spectorgram(path_list,
                                                     n_mels=param["feature"]["n_mels"],
                                                     frames=None,
                                                     )

                        tf.summary.image(f'{type} true sepctrogram',
                                         numpy.asarray(spec_fig_list), step=0, max_outputs=10)
                        tf.summary.image(f'{type} predict sepctrogram',
                                         numpy.asarray(pred_spec_fig_list), step=0, max_outputs=10)

                        clean_spec_fig_list = evaluation.\
                            draw_clean_spectrogram(path_list,
                                                   clean_dir=param['clean_directory'],
                                                   type=type,
                                                   n_mels=param["feature"]["n_mels"],
                                                   n_fft=param["feature"]["n_fft"],
                                                   hop_length=param["feature"]["hop_length"],
                                                   )
                        tf.summary.image(f'{type} clean sepctrogram',
                                         numpy.asarray(clean_spec_fig_list), step=0, max_outputs=10)

                else:
                    file_path_dict = evaluation.select_data(reconst_score,
                                                            reconst_dict)

                    for type, path_list in file_path_dict.items():
                        spec_fig_list = evaluation. \
                            draw_true_spectrogram(path_list,
                                                  n_mels=param["feature"]["n_mels"],
                                                  n_fft=param["feature"]["n_fft"],
                                                  hop_length=param["feature"]["hop_length"],
                                                  )
                        pred_spec_fig_list = evaluation. \
                            draw_predict_spectorgram(path_list,
                                                     n_mels=param["feature"]["n_mels"],
                                                     frames=None,
                                                     )

                        tf.summary.image(f'{type} true sepctrogram',
                                         numpy.asarray(spec_fig_list), step=0, max_outputs=10)
                        tf.summary.image(f'{type} predict sepctrogram',
                                         numpy.asarray(pred_spec_fig_list), step=0, max_outputs=10)

    if mode:
        # output results
        result_path = "{result}/{file_name}".format(result=param["result_directory"],
                                                    file_name=param["result_file"])
        com.logger.info("AUC and pAUC results -> {}".format(result_path))
        save_csv(save_file_path=result_path, save_data=csv_lines)
