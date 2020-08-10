"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
from tqdm import tqdm
import common_cnn_n as com
import model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path
from keras import backend as K
from keras.losses import mse, binary_crossentropy
import random
from keras.models import Model
import wavenet
from keras import metrics
from keras import objectives
from wavenet_utils import CausalAtrousConvolution1D, categorical_mean_squared_error
import keras

########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


########################################################################
# select gpu
########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
########################################################################


def list_to_vector_array(file_list,
                         labels,
                         msg="calc...",
                         n_mels=64,
                         num_ch=1,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         frames=None):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
    """
    # iterate file_to_vector_array()
    dataset_list = []
    data_dataset_list = []

    file_list = numpy.asarray(file_list)
    data_file_list = file_list[labels == 1]
    noise_file_list = file_list[labels == 0]

    if param['denoising'] == 'withnoise':
        datalist = file_list
        noiselist = file_list
    else:
        datalist = data_file_list
        noiselist = noise_file_list

    for idx in tqdm(range(len(datalist)), desc=msg):
        vector_array, data_vector_array = com.file_to_vector_array(datalist[idx],
                                                                   noiselist[idx],
                                                                   num_ch=num_ch,
                                                                   n_mels=n_mels,
                                                                   n_fft=n_fft,
                                                                   hop_length=hop_length,
                                                                   power=power,
                                                                   frames=None)

        dataset_list.append(vector_array)
        data_dataset_list.append(data_vector_array)

    dataset_np = numpy.asarray(dataset_list)
    data_dataset_np = numpy.asarray(data_dataset_list)

    if len(dataset_np.shape) == 3:  # vector_array shape : B, dims (for 1D FC input)
        dataset = dataset_np.reshape(
            (vector_array.shape[0] * len(file_list), vector_array.shape[1]))
        data_dataset = data_dataset_np.reshape(
            (vector_array.shape[0] * len(file_list), vector_array.shape[1]))
        # B, dims

    elif len(dataset_np.shape) == 4:  # vector_array shape : C, F, T (for 2D CNN input)
        dataset = numpy.moveaxis(dataset_np, 1, 3)  # B, F, T, C
        data_dataset = numpy.moveaxis(data_dataset_np, 1, 3)  # B, T, F

    else:
        com.logger.exception('not expected vector array dimension!')
        sys.exit(-1)

    return dataset, data_dataset


def file_list_generator(target_dir,
                        select=None,
                        dir_name="train",
                        ext="wav",
                        ):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_data_list_path = os.path.abspath(f"{target_dir}/{dir_name}/normal/*.{ext}")
    training_noise_list_path = os.path.abspath(f"{target_dir}/{dir_name}/noise/*.{ext}")

    train_data_files = sorted(glob.glob(training_data_list_path))
    noise_files = sorted(glob.glob(training_noise_list_path))

    # random select if needed
    if type(select) == int:
        train_data_files = random.sample(train_data_files, select)
        noise_files = random.sample(noise_files, select)

    train_data_labels = numpy.ones(len(train_data_files), dtype=numpy.int8)
    noise_labels = numpy.zeros(len(noise_files), dtype=numpy.int8)

    files = numpy.concatenate((train_data_files, noise_files), axis=0)
    labels = numpy.concatenate((train_data_labels, noise_labels), axis=0)

    if len(files) == 0:
        com.logger.exception(f"no_{ext}_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))

    return files, labels


def vae_loss(inputs, outputs, z_mean, z_log_var, original_dim, reconst_loss='mse'):
    if reconst_loss == 'mse':
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    return vae_loss


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

########################################################################


########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["data_directory"], exist_ok=True)

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    # set writer
    # writer = tf.compat.v1.summary.FileWriter(param['log_dir'])

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                     machine_type=machine_type)
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                  machine_type=machine_type)

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue

        # generate dataset
        print("============== DATASET_GENERATOR ==============")

        data_path = Path(f'{param["data_directory"]}')

        if os.path.exists(data_path/'org_data.npy'):  # if data exists
            com.logger.info("data exists")
            noisy_data = numpy.load(data_path/'noisy_data.npy')
            org_data = numpy.load(data_path / 'org_data.npy')
            labels = numpy.load(data_path/'labels.npy')

        else:  # if data do not exists
            files, labels = file_list_generator(target_dir, ext='wav', select=None)
            noisy_data, org_data = list_to_vector_array(files,
                                                        labels,  # 1 for data, 0 for noise
                                                        msg="generate train_dataset",
                                                        **param['feature'])

            # save data
            numpy.save(data_path / 'noisy_data.npy', noisy_data)  # (B, D) or (B, F, T, C)
            numpy.save(data_path / 'org_data.npy', org_data)  # (B, D) or (B, F, T)
            numpy.save(data_path / 'labels.npy', labels)

        print("============== MODEL TRAINING ==============")
        org_data_mean = numpy.mean(org_data, axis=(0, 1))
        org_data_std = numpy.std(org_data, axis=(0, 1))
        numpy.save(data_path / 'org_data_mean.npy', org_data_mean)  # (B, D) or (B, F, T, C)
        numpy.save(data_path / 'org_data_std.npy', org_data_std)

        print('---standardization---')
        for i in tqdm(range(org_data.shape[1])):
            org_data[:, i, :] = (org_data[:, i, :] - org_data_mean[i]) / org_data_std[i]

        epochs = param['fit']['epochs']
        batch_size = param['fit']['batch_size']

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=param['log_dir'],
                                                     batch_size=batch_size,
                                                     write_graph=True,
                                                     write_grads=True)

        org_data = org_data.squeeze()
        org_data = numpy.moveaxis(org_data, 1, 2)  # B, T, F

        # Model Training
        input_size = org_data.shape[1]
        num_channel = org_data.shape[2]
        num_filters = 5 * num_channel
        kernel_size = 3
        num_residual_blocks = 3

        model = model.build_wavenet_model(input_size, num_channel, num_filters,
                                          kernel_size, num_residual_blocks)

        mc = keras.callbacks.ModelCheckpoint(param['model_directory']
                                             + '/weights.{epoch:08d}.h5',
                                             save_weights_only=False, period=10)

        history = model.fit(org_data,
                            org_data[:, 8:, :],
                            **param['fit'],
                            callbacks=[tensorboard, mc])

        # model = wavenet.build_model(fragment_length=org_data.shape[1])
        #
        # optim = wavenet.make_optimizer()
        # loss = objectives.categorical_crossentropy
        # all_metrics = [metrics.categorical_accuracy,
        #                categorical_mean_squared_error]
        #
        # model.compile(optimizer=optim, loss=loss, metrics=all_metrics)
        #
        # model.fit(org_data,
        #           org_data,
        #           **param['fit'],
        #           callbacks=[tensorboard]
        #           )

        model.save(model_file_path + '_model.hdf5')
        com.logger.info("save_model -> {}".format(model_file_path))
        print("============== END TRAINING ==============")



