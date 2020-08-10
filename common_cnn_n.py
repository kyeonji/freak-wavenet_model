"""
 @file   common.py
 @brief  Commonly used script
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys
import os

# additional
import numpy
import librosa
import librosa.core
import librosa.feature
import yaml
from keras.models import Model
from keras.layers import Input
from keras import optimizers

########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.0"
########################################################################

########################################################################
# argparse
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-v', '--version', action='store_true', help="show application version")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("DCASE 2020 task 2 baseline\nversion {}".format(__versions__))
        print("===============================\n")
    if args.eval ^ args.dev:
        if args.dev:
            flag = True
        else:
            flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag
########################################################################


########################################################################
# load parameter.yaml
########################################################################
def yaml_load():
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param


param = yaml_load()
########################################################################


########################################################################
# file I/O
########################################################################
# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################


########################################################################
# feature extractor
########################################################################
def demux_wav(wav_name, channel=1):
    """
    demux .wav file.
    wav_name : str
        target .wav file
    channel : int
        target channel number
    return : numpy.array( float )
        demuxed mono data
    Enabled to read multiple sampling rates.
    Enabled even one channel.
    """
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data

        return sr, numpy.array(multi_channel_data)[:channel, :]

    except ValueError as msg:
        logger.warning(f'{msg}')


def file_to_vector_array(data_file_name,
                         noise_file_name,
                         num_ch=1,
                         frames=None,
                         n_mels=64,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         ):
    """
    convert file_name to a vector array.
    file_name : str
        target .wav file
    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 generate melspectrogram using librosa (**kwargs == param["librosa"])
    mel_spec_list = []
    data_mel_spec_list = []
    data_sr, data_ys = demux_wav(data_file_name, num_ch)  # ys : c*t

    if noise_file_name == None:
        noise_ys = numpy.zeros_like(data_ys)
    else:
        noise_sr, noise_ys = demux_wav(noise_file_name, num_ch)

    for channel in range(num_ch):  # multi channel input
        try:
            data_y = data_ys[channel, :]
            noise_y = noise_ys[channel, :]
            y = data_y + noise_y
        except:
            y = data_ys + noise_ys
            data_y = data_ys

        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                         sr=data_sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_mels=n_mels,
                                                         power=power)

        data_mel_spectrogram = librosa.feature.melspectrogram(y=data_y,
                                                              sr=data_sr,
                                                              n_fft=n_fft,
                                                              hop_length=hop_length,
                                                              n_mels=n_mels,
                                                              power=power)

        # 02 convert melspectrogram to log mel energy
        log_mel_spectrogram = librosa.core.power_to_db(mel_spectrogram)
        data_log_mel_spectrogram = librosa.core.power_to_db(data_mel_spectrogram)

        # 03 list up multi channel spectrogram
        mel_spec_list.append(log_mel_spectrogram)
        data_mel_spec_list.append(data_log_mel_spectrogram)

    mel_spec_np = numpy.asarray(mel_spec_list)
    data_mel_spec_np = numpy.asarray(data_mel_spec_list)

    return mel_spec_np, data_mel_spec_np  # 1, 128, 344


# load dataset
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        dir_path = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
        dirs = sorted(glob.glob(dir_path))
    else:
        logger.info("load_directory <- evaluation")
        dir_path = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
        dirs = sorted(glob.glob(dir_path))
    return dirs


def get_all_network(denoising_model, ad_model, train_data):
    ad_model.trainable = False

    inputDim = train_data[0].shape  # dims
    x_in = Input(shape=inputDim)

    x = denoising_model(x_in)

    net_output = ad_model(x)

    net = Model(inputs=x_in, outputs=net_output)
    net.compile(**param['compile'])
    return net


import numpy as np
import os


def mu_law(xt, mu=255):
    """ Transforms the normalized audio signal to values between [-1, 1]
        so that it can be quantized in range [0, 255] for the softmax output.
        See Section 2.2 of the paper [1].
        See:
            [1] Oord, Aaron van den, et al. "Wavenet: A generative model for
                raw audio." arXiv preprint arXiv:1609.03499 (2016).
    """
    return np.sign(xt) * (np.log(1 + mu * np.absolute(xt)) / np.log(1 + mu))


def mu_law_inverse(yt, mu=255):
    """ The inverse transformation of mu-law that expands the input back to
        the original space.
    """
    return np.sign(yt) * (1 / mu) * (((1 + mu) ** np.abs(yt)) - 1)


def get_audio_sample_batches(vector_array, receptive_field_size,
                             stride_step=1):
    """ Provides the audio data in batches for training and validation.
        Note: This function used to be a generator, but when experimenting
        with little data, a function makes more sense.
        Args:
            path_to_audio_train (str): Path to the directory containin the
                audio files for training.
            receptive_field_size (int): The size of the sliding window that
                passes over the data and collects training samples.
            stride_step (int, default:32): The step by which the window slides.
    """
    X = []
    y = []

    offset = 0
    while offset + receptive_field_size < vector_array.shape[2]:
        X.append(vector_array[:, :, offset:offset + receptive_field_size])
        y.append(vector_array[:, :, offset + receptive_field_size])
        offset += stride_step
    X = np.array(X)
    y = np.array(y)

    return X, y
########################################################################

