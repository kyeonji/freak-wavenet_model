import scikitplot as skplt
import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
import sklearn
from sklearn.metrics import precision_recall_curve
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
import librosa


def fig2rgb_array(fig, expand=True):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def draw_confusion_mat(y, y_est, threshold, show=False):
    y = np.asarray(y)
    label_pred = []
    for score in y_est:
        label_pred.append(int(score > threshold))

    fig, ax = plt.subplots(dpi=150)

    skplt.metrics.plot_confusion_matrix(y, label_pred, ax=ax, normalize=True)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    if show:
        fig.show()

    return fig2rgb_array(fig)


def draw_roc_curve(y, y_pred, show=False):
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    stack_y_pred = np.stack((50 - y_pred, y_pred), axis=1)

    fig, ax = plt.subplots(dpi=150)

    skplt.metrics.plot_roc(y, stack_y_pred, ax=ax,
                           plot_macro=False, plot_micro=False, classes_to_plot=1)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    if show:
        fig.show()

    return fig2rgb_array(fig)


def plot_hist(y_true, y_pred, log):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_data_pred = y_pred[:len(y_true)]
    y_noise_pred = y_pred[len(y_true):]

    fig, ax = plt.subplots(dpi=150)
    bins = np.linspace(min(y_pred), max(y_pred), 100)
    # bins = np.linspace(0, 50, 100)
    ax.hist(y_data_pred[y_true == 0], bins, alpha=0.5, log=log, label='NORMAL')
    ax.hist(y_data_pred[y_true == 1], bins, alpha=0.5, log=log, label='ANOMALY')
    ax.hist(y_noise_pred, bins, alpha=0.5, log=log, label='NOISE')
    fig.legend(loc='upper right')

    return fig2rgb_array(fig)


def plot_multi_hist(y_true, y_pred1, y_pred2,  log):
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    y_data_pred1 = y_pred1[:len(y_true)]
    y_noise_pred1 = y_pred1[len(y_true):]
    y_data_pred2 = y_pred2[:len(y_true)]
    y_noise_pred2 = y_pred2[len(y_true):]

    fig, ax = plt.subplots(dpi=150)
    bins = np.linspace(np.min([y_pred1, y_pred2]), np.max([y_pred1, y_pred2]), 100)
    # bins = np.linspace(0, 100, 300)
    ax.hist(y_data_pred1[y_true == 0], bins, alpha=0.5, log=log, label='NOISE NORMAL')
    ax.hist(y_data_pred1[y_true == 1], bins, alpha=0.5, log=log, label='NOISE ANOMALY')
    ax.hist(y_noise_pred1, bins, alpha=0.5, log=log, label='NOISE NOISE')
    ax.hist(y_data_pred2[y_true == 0], bins, alpha=0.5, log=log, label='DENOISE NORMAL')
    ax.hist(y_data_pred2[y_true == 1], bins, alpha=0.5, log=log, label='DENOISE ANOMALY')
    ax.hist(y_noise_pred2, bins, alpha=0.5, log=log, label='DENOISE NOISE')
    fig.legend(loc='upper right')

    return fig2rgb_array(fig)


def draw_PRcurve(y_true, y_pred):
    fig, ax = plt.subplots(dpi=150)

    precision, recall, threshold = precision_recall_curve(y_true, y_pred)
    auc = sklearn.metrics.auc(recall, precision)
    plt.plot(recall, precision, scalex=True, scaley=False, marker='.', label='Logistic')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.title(f'AUC score :{auc}')

    return fig2rgb_array(fig)


def cal_f1score(y_true, y_pred, threshold):
    label_pred = []
    for score in y_pred:
        label_pred.append(int(score > threshold))

    f1_score = sklearn.metrics.f1_score(y_true, label_pred,
                                        labels=None,
                                        pos_label=1,
                                        average='binary',
                                        sample_weight=None,
                                        )

    return f1_score


def recostruct_spectrogram(predict,
                           n_mels,
                           frames,
                           ):

    if len(predict.shape) == 2:  # (B, D)
        time = frames - 1 + predict.shape[0]
        spec = np.zeros((n_mels, time))
        for B_id in range(predict.shape[0]):
            if B_id == 0:
                for D_id in range(frames):
                    spec[:, D_id] = predict[B_id, n_mels * D_id:n_mels * (D_id + 1)].T
            else:
                D_id = frames - 1
                spec[:, B_id + D_id] = predict[B_id, n_mels * D_id:n_mels * (D_id + 1)].T

    elif len(predict.shape) == 4:  # (1, F, T, C)
        predict = predict.squeeze(axis=0)
        spec = predict[:, :, 0]

    else:
        predict = predict.squeeze(axis=0)
        spec = np.moveaxis(predict, 0, 1)

    return spec


def write_audio(file_path_list):

    audio_list = []
    for file_path in file_path_list:
        audio, sr = librosa.load(file_path[0], sr=None)
        audio_list.append(audio)

    return audio_list, sr


def draw_true_spectrogram(file_path_list,
                          n_fft,
                          hop_length,
                          n_mels,
                          power=2,
                          ):

    spec_fig_list = []
    for file_path in file_path_list:
        # file_path = [file_path, score]
        y, sr = librosa.load(f'{file_path[0]}', sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_mels=n_mels,
                                                         power=power)
        log_mel_spectrogram = librosa.core.power_to_db(mel_spectrogram[:, 40:264])

        log_mel_spec_fig = plt.figure(dpi=150, )
        plt.imshow(log_mel_spectrogram,
                   cmap=plt.get_cmap('CMRmap'),
                   origin='lower', aspect='auto')
        plt.xlabel('Frame Index')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'score:{file_path[2]}')
        plt.colorbar()
        log_mel_spec_fig = fig2rgb_array(log_mel_spec_fig).squeeze()
        spec_fig_list.append(log_mel_spec_fig)

    return spec_fig_list


def draw_predict_spectorgram(file_path_list,
                             n_mels,
                             frames):
    predict_sepc_fig_list = []
    for file_path in file_path_list:
        # file_path = [file_path, score]
        predcit = file_path[1]
        predict_spectrogram = recostruct_spectrogram(predcit,
                                                     n_mels=n_mels,
                                                     frames=frames)
        predict_spec_fig = plt.figure(dpi=150, )
        plt.imshow(predict_spectrogram,
                   cmap=plt.get_cmap('CMRmap'),
                   origin='lower', aspect='auto')
        plt.xlabel('Frame Index')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'score:{file_path[2]}')
        plt.colorbar()
        predict_spec_fig = fig2rgb_array(predict_spec_fig).squeeze()
        predict_sepc_fig_list.append(predict_spec_fig)

    return predict_sepc_fig_list


def draw_clean_spectrogram(file_path_list,
                           clean_dir,
                           type,
                           n_fft,
                           hop_length,
                           n_mels,
                           power=2):
    spec_fig_list = []
    for file_path in file_path_list:
        file_name = file_path[0].split('/')[-1]
        y, sr = librosa.load(f'{clean_dir}/{type}/{file_name}', sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_mels=n_mels,
                                                         power=power)
        log_mel_spectrogram = librosa.core.power_to_db(mel_spectrogram[:, 40:264])

        log_mel_spec_fig = plt.figure(dpi=150, )
        plt.imshow(log_mel_spectrogram,
                   cmap=plt.get_cmap('CMRmap'),
                   origin='lower', aspect='auto')
        plt.xlabel('Frame Index')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'score:{file_path[2]}')
        plt.colorbar()
        log_mel_spec_fig = fig2rgb_array(log_mel_spec_fig).squeeze()
        spec_fig_list.append(log_mel_spec_fig)

    return spec_fig_list


def select_data(denoising_score,
                denoised_dict,
                ):

    normal_list = []
    anomaly_list = []
    for i, (path, denoised) in enumerate(denoised_dict.items()):
        file_name = path.split('/')[-1]
        if 'normal' in file_name.split('_') or 'O' in path.split('_'):
            normal_list.append([path, denoised, denoising_score[i]])
        else:
            anomaly_list.append([path, denoised, denoising_score[i]])

    sorted_normal = sorted(normal_list, key=lambda x: x[2])
    sorted_anomaly = sorted(anomaly_list, key=lambda x: x[2])

    for list in [sorted_normal, sorted_anomaly]:
        total_score = 0
        for data in list:
            total_score += float(data[2])
        avg_score = total_score / len(list)
        score_list = []
        for data in list:
            score_list.append(float(data[2])-avg_score)
        mean_file_index = score_list.index(min(score_list))

    normal_path_list = [sorted_normal[0], sorted_normal[1],
                        sorted_normal[mean_file_index], sorted_normal[len(sorted_normal)//2],
                        sorted_normal[-2], sorted_normal[-1]]
    anomaly_path_list = [sorted_anomaly[0], sorted_anomaly[1],
                        sorted_anomaly[mean_file_index], sorted_anomaly[len(sorted_anomaly)//2],
                        sorted_anomaly[-2], sorted_anomaly[-1]]

    file_path_dict = {'normal': normal_path_list, 'anomaly': anomaly_path_list}

    return file_path_dict
