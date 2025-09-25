import tkinter as tk
from tkinter import filedialog

import os
import numpy as np
import mne
from scipy import stats
import copy
import re
import math
import pickle
import matplotlib.pyplot as plt
import sys 


from pydub import AudioSegment
from sklearn.model_selection import KFold
from spyeeg.models.TRF import TRFEstimator

from collections import Counter



def vhdr_to_mne(in_path):
    raw_fif = []
    for fname in in_path:
        print(fname)
        path, file = os.path.split(fname)
        print(file)
        # Find name of the VHDR file inside the subject's directory
        f, ext = os.path.splitext(file)
        if ext == '.vhdr':
            # Convert to .fif format
            raw_fif.append(
    mne.io.read_raw_brainvision(os.path.join(path, file), preload=True, verbose='ERROR')
)

    return raw_fif

def bipolar_montage(raw, chans):
    channels = raw.info['ch_names'] # store ch_names
    raw_mono = raw.get_data().T     # fif to np

    raw_bipol = []
    channels_bipol = []

    # si ce qui est avant le chiffre est =, alors soustrait
    for i in range(len(channels) - 1):
        electrode1 = re.split(r'\d', channels[i])[0]
        electrode2 = re.split(r'\d', channels[i+1])[0]
        if electrode1 == electrode2:
            activity_diff = raw_mono[:, i+1] - raw_mono[:, i]
            channels_bipol.append(f'{channels[i]}-{channels[i+1]}')
            raw_bipol.append(activity_diff)

    raw_bipol_list = [arr.tolist() for arr in raw_bipol]
    raw_bipol_array = np.vstack(raw_bipol_list)

    if chans == True :
        return raw_bipol_array.T, channels_bipol
    if chans == False :
        return raw_bipol_array.T

def prepare_data(data, picks, high_pass, low_pass, fs_data) :

    raw_filtered = np.zeros((len(data), len(picks)))

    # Filtrer les données
    for i_chan in range(len(picks)):
        raw_filtered[:,i_chan] = mne.filter.filter_data(data=data[:,i_chan], sfreq=fs_data, l_freq=high_pass, h_freq=low_pass, verbose=False)

    # Downsampler les données
    new_fs = 100
    raw_downsampled = mne.filter.resample(raw_filtered, down=fs_data/new_fs, npad='auto', axis=0)    # 100Hz
    raw_downsampled = np.expand_dims(raw_downsampled, axis=1)

    # z-scoring
    raw_zscored = stats.zscore(raw_downsampled,axis=0)

    return raw_zscored

def concatenation(pianos, data_path, stim_keeped):

    #out_path for concatenated

    regressors_path = os.path.join(data_path, 'stim', 'reg')


    pianos_cut = []
    acoustic_regressors_keeped = []
    wd_regressors_keeped = []

    for i, piano in enumerate(pianos) :

        stim_name = stim_keeped[i].rsplit('.')[0]

        ## loading acoustic regressors
        with open(regressors_path + f'/acoustic_regs_{stim_name}.pkl', 'rb') as file:
            acoustic_regressors = pickle.load(file)

        ## loading predictive regressors
        with open(regressors_path + f'/predictive_regs_{stim_name}.pkl', 'rb') as file:
            wd_regressors = pickle.load(file)

        # cut Y
        acoustic_regressors['X'] = acoustic_regressors['X'][:piano.shape[0],:,:]
        wd_regressors['X'] = wd_regressors['X'][:piano.shape[0],:,:]

        piano = piano[:acoustic_regressors['X'].shape[0],:,:]

        # store
        acoustic_regressors_keeped.append(acoustic_regressors['X'])
        wd_regressors_keeped.append(wd_regressors['X'])

        pianos_cut.append(piano)


    # concatenate Y
    Y = np.concatenate(pianos_cut, axis=0)

    # concatenate regs
    acoustic_regressors_concat = np.concatenate(acoustic_regressors_keeped, axis=0)
    wd_regressors_concat = np.concatenate(wd_regressors_keeped, axis=0)

    X = np.concatenate((acoustic_regressors_concat, wd_regressors_concat), axis = 2)
    labels = acoustic_regressors['label']+wd_regressors['label']
    fs = acoustic_regressors['fs']


    return Y, X, labels, fs


def load_laure_sorciere_regressors(data_path, data):
    """
    Charge tous les régressseurs (acoustiques et prédictifs),
    applique un resampling à 100 Hz sur les régressseurs acoustiques (z-score déjà fait),
    applique un z-score sur les prédictifs,
    concatène l’ensemble et aligne avec les données EEG.

    Paramètres :
        data_path : chemin vers les fichiers de régressseurs
        data : données EEG à aligner temporellement

    Retour :
        data : EEG tronqué
        X : array (n_time x n_subjects x n_regresseurs)
        labels : liste des noms des régressseurs
    """

    regressors_path = os.path.join(data_path, 'stim', 'reg')

    # --- Charger les régressseurs acoustiques (non z-scored) ---
    with open(os.path.join(regressors_path, 'All_Regressors_not_zscored.pkl'), 'rb') as file:
        acoustic_regressors = pickle.load(file)

    reg_data_all = np.array(acoustic_regressors['regressors']) 
    reg_names_all = acoustic_regressors['name'] 

    fs_old = 500
    fs_new = 100
    n_time_new = int(reg_data_all.shape[0] * fs_new / fs_old)

    # Resample à 100 Hz
    X_acoustic_resampled = resample(reg_data_all, n_time_new, axis=0)

    # Z-score des régressseurs acoustiques
    X_acoustic_resampled = stats.zscore(X_acoustic_resampled, axis=0)

    # --- Charger les régressseurs prédictifs ---
    with open(os.path.join(regressors_path, 'predictive_regressors_wd.pkl'), 'rb') as file:
        wd_regressors = pickle.load(file)

    # Réduction aléatoire à même longueur
    idx_to_keep = sorted(np.random.choice(
        wd_regressors['X'].shape[0],
        X_acoustic_resampled.shape[0],
        replace=False
    ))
    wd_X = wd_regressors['X'][idx_to_keep, :, :]  # shape: (n_time_new, n_subjects, n_predictive)

    # Z-score des prédictifs (précaution)
    wd_X = stats.zscore(wd_X, axis=0)

    # --- Concaténer ---
    X = np.concatenate((X_acoustic_resampled, wd_X), axis=2)
    labels = reg_names_all + wd_regressors['names']

    # --- Tronquer les données EEG ---
    n_min = min(len(data), X.shape[0])
    return data[:n_min], X[:n_min], labels

def load_sorciere_regressors(data_path, data) :

    regressors_path = os.path.join(data_path, 'stim', 'reg')

    with open(regressors_path +'/acoustic_regressors.pkl', 'rb') as file:
        acoustic_regressors = pickle.load(file)

    with open(regressors_path +'/predictive_regressors_wd.pkl', 'rb') as file:
        wd_regressors = pickle.load(file)

    ## there are 20 timepoints in excess in pred_dict
    idx_to_keep = sorted(np.random.choice(wd_regressors['X'].shape[0], acoustic_regressors['X'].shape[0], replace=False))

    ## we randomly select 20 points to drop
    wd_regressors['X'] = wd_regressors['X'][idx_to_keep,:,:]
    wd_regressors['X'] = stats.zscore(wd_regressors['X'], axis=0)

    ## define reg and labels
    X = np.concatenate((acoustic_regressors['X'], wd_regressors['X']), axis = 2)
    X = X[:len(data),:,:]
    data = data[:len(X),:,:]

    labels = acoustic_regressors['names']+wd_regressors['names']

    return data, X, labels

def model_definition(regs_all, regs_all_label, model_acoustic_names, model_predictif_names):
    """
    Sélectionne les régressseurs acoustiques et prédictifs à partir de leurs noms.

    Args:
        regs_all (np.ndarray): Matrice des régressseurs.
        regs_all_label (list of str): Liste des labels pour tous les régressseurs.
        model_acoustic_names (list of str): Liste des noms de régressseurs acoustiques à garder.
        model_predictif_names (list of str): Liste des noms de régressseurs prédictifs à garder.

    Returns:
        regs_acoustic, regs_acoustic_labels, regs_predictive, regs_predictive_labels
    """
    # Construction d'un dictionnaire {label: index}
    label_to_index = {label: idx for idx, label in enumerate(regs_all_label)}

    # Sélection via les noms
    acoustic_idx = [label_to_index[name] for name in model_acoustic_names]
    predictif_idx = [label_to_index[name] for name in model_predictif_names]

    # Extraction
    regs_acoustic = regs_all[:,:,acoustic_idx]
    regs_acoustic_labels = model_acoustic_names

    regs_predictive = regs_all[:,:,predictif_idx]
    regs_predictive_labels = model_predictif_names

    return regs_acoustic, regs_acoustic_labels, regs_predictive, regs_predictive_labels

def trf_crossval(regressor, data, fs, picks, reg_labels, tw_trf, n_folds, alphas):

    X_ = copy.deepcopy(regressor)
    Y_ = copy.deepcopy(data)

    chunk_size = np.floor(X_.shape[0]/n_folds).astype(int)
    # init the chunk with the right format
    tmp_X = np.full( (chunk_size,n_folds, X_.shape[2]), None)
    tmp_Y = np.full( (chunk_size,n_folds, Y_.shape[2]), None)
    # fill the chunks
    for i_fold in range(n_folds):
        tmp_X[:,i_fold,:] = X_[i_fold*chunk_size:(i_fold+1)*chunk_size, 0, :]
        tmp_Y[:,i_fold,:] = Y_[i_fold*chunk_size:(i_fold+1)*chunk_size, 0, :]
    X_, Y_ = tmp_X.astype(float), tmp_Y.astype(float)

    #some parameters
    scoring = 'r2'
    alphas = alphas

    # Time windows for the response function
    times = np.linspace(tw_trf[0],tw_trf[1], int(np.diff(tw_trf)*fs +1))    #get TRF time-axis

    # set up cross-validation scheme
    cv = KFold(n_folds, shuffle=True, random_state=37)  # should try with StratifiedKFold

    #some variables to store the results
    models = [[] for _ in range(len(alphas))]                             #store all models
    best_alphas = [[] for _ in range(len(picks))]                         #best alpha per channel

    scores_cv = np.full(len(picks),np.nan)                                #cross-validated score          (with the best alpha)
    coefs_cv = np.full([len(picks),X_.shape[2],times.shape[0]],np.nan)     #averaged TRF shape across cv   (with the best alpha)

    scores_fit = np.full(len(picks),np.nan)                               #score of global fit            (with the best alpha)
    coefs_fit = np.full([len(picks),X_.shape[2],times.shape[0]],np.nan)    #TRF shape of global fit        (with the best alpha)
    predicted_Y = np.zeros_like(Y_)                                        #predicted Y of global fit      (with the best alpha)

    #for each channel
    for i_chan in range(len(picks)):
        print('\nChannel : ', picks[i_chan])
        Y_chan = np.expand_dims(Y_[:,:,i_chan],2)
        scores_val = np.zeros_like(alphas)                #store validation scores for each alpha

        #for each alpha
        for ii, alpha in enumerate(alphas):
            # print('alpha:', alpha)
            in_score = np.zeros(n_folds)

            #for each cv
            for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X_, 1, 0))):
                #define model
                rf = mne.decoding.ReceptiveField(tw_trf[0], tw_trf[1], fs, feature_names=reg_labels, estimator=alpha, scoring=scoring, n_jobs=-1,verbose=False)

                #fit on train data
                rf.fit(X_[:, train, :], Y_chan[:, train, :])

                #store score & model at each cv
                in_score[fold_number] = np.mean(rf.score(X_[:, test, :], Y_chan[:, test, :]))
                models[ii].append(rf)

            #get averaged score for this alpha, across cv
            scores_val[ii] = in_score.mean()
            # print('CV_Score:', scores_val[ii])

        # Choose the model that performed best on test data
        ix_best_alpha = np.argmax(scores_val)
        best_alphas[i_chan] = alphas[ix_best_alpha] #retrieve best alpha for this channel
        print('best score', str(scores_val[ix_best_alpha]), ' obtained with alpha = ', str(best_alphas[i_chan]))

        #store averaged coefs and scores for best alpha
        coefs_cv[i_chan,:,:] = np.array([models[ix_best_alpha][icv].coef_ for icv in range(n_folds)]).mean(axis=0)
        scores_cv[i_chan] = scores_val[ix_best_alpha]

        #fit another model on the entire dataset, with the best alpha
        if not math.isnan(np.max(scores_val)):
            rf = mne.decoding.ReceptiveField(tw_trf[0], tw_trf[1], fs, feature_names=reg_labels, estimator=best_alphas[i_chan], scoring=scoring, n_jobs=-1, verbose=False)
            rf.fit(X_, Y_chan)
            coefs_fit[i_chan,:,:] = rf.coef_
            scores_fit[i_chan] = rf.score(X_, Y_chan)
            predicted_Y[:,:,i_chan] = np.squeeze(rf.predict(X_))
            print('score fit :', scores_fit[i_chan])

    return scores_cv, coefs_fit, best_alphas


#LEGACY
# def trf_spyeeg(regs, data, tmin, tmax, fs, alpha, channels):

#     trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=fs, alpha=alpha, fit_domain='frequency')

#     #shape(n_folds, n_channels, n_alpha)
#     scores = trf.xval_eval(regs, data, segment_length=None, fit_mode='direct', verbose = False)
#     #shape (n_chan, n_alpha)
#     scores = np.mean(scores, axis=0)
#     #shape (tw, n_reg, n_chan, n_alpha)
#     kernels = trf.get_coef()

#     ix_best_alpha = np.argmax(scores, axis=1)

#     #shape (n_chan)
#     scores_ = np.zeros(len(channels))
#     #shape (tw, n_reg, n_chan)
#     kernels_ = np.zeros((kernels.shape[0], kernels.shape[1], kernels.shape[2]))

#     for i_chan in range(len(channels)):
#         scores_[i_chan] = scores[i_chan, ix_best_alpha[i_chan]]
#         kernels_[:,:,i_chan] = kernels[:,:,i_chan, ix_best_alpha[i_chan]]
#         filled_len = int(20 * (i_chan + 1) / len(channels))
#         bar = '█' * filled_len + '-' * (20 - filled_len)
#         sys.stdout.write(f'\r  [{bar}] {i_chan + 1}/{len(channels)}')
#         sys.stdout.flush()

#     return scores_**2, kernels_



def histograms_results(cat, acoustic_scores, predictive_scores, picks, out_path, fig_name) :

    group_indices = []
    group_label = []

    for i, elec in enumerate(picks):
        label = re.split(r'\d', elec, maxsplit=1)[0]
        if label != re.split(r'\d', picks[i - 1], maxsplit=1)[0]:
            group_label.append(label)
            group_indices.append(i)
        if i == len(picks) - 1:
            group_indices.append(i)

    fig, (ax0, ax1) = plt.subplots(2,1)

    ## first : no comparaison just simpl_acoustic
    ax0.bar(x=picks, height=(100 * acoustic_scores).tolist())
    if cat == 'Speech':
        ax0.set_title('Encoding of acoustic information (speech)')  # Annotation text
    else :
        ax0.set_title('Encoding of acoustic information (music)')  # Annotation text

    ax0.set_xticks(picks)
    ax0.set_xticklabels(picks, rotation=45, ha='right', fontsize=8)  # Rotation et alignement ajustés

    ax0.set_ylim(0, np.max(100 * acoustic_scores) + .25)

    ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    for i, label in enumerate(group_label):
        ax0.axvline(x=group_indices[i] - 0.5, color='gray', linestyle='--', linewidth='.5', alpha=.7)
        ax0.text(group_indices[i] + ((group_indices[i + 1] - group_indices[i]) / 2),
                 (np.max(100 * acoustic_scores) + .25) / 2, label)

    ax1.bar(picks, (100 * (predictive_scores - acoustic_scores)).tolist())
    if cat == 'Speech':
        ax1.set_title('Encoding of predictive regions (speech)')
    else :
        ax1.set_title('Encoding of predictive regions (music)')
    ax1.set_xticks(picks)
    ax1.set_xticklabels(picks, rotation=45, ha='right', fontsize=8)  # Rotation et alignement ajustés

    ax1.set_ylim(0, np.max(100 * acoustic_scores) + .25)

    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    for i, label in enumerate(group_label):
        ax1.axvline(x=group_indices[i] - 0.5, color='gray', linestyle='--', linewidth='.5', alpha=.7)
        ax1.text(group_indices[i] + ((group_indices[i + 1] - group_indices[i]) / 2),
                 (np.max(100 * (predictive_scores - acoustic_scores)) + .25) / 2, label)
    ax1.set_ylim(0, np.max(100 * (predictive_scores - acoustic_scores) + .25))



    plt.savefig(out_path + fig_name)
    plt.show()
    
def align(raw_fif, raw_array, stim_dir, offset):
    # --- Load and extract trigger from audio ---
    aud = AudioSegment.from_file(stim_dir, format="mp3")
    print("\n" + "=" * 60)
    print(f"🎧 Alignement de : {os.path.basename(stim_dir)}")
    print("=" * 60)
    fs_stim = aud.frame_rate
    trigger_channel = np.array(aud.split_to_mono()[1].get_array_of_samples())

    # --- Detect trigger onset in audio using threshold ---
    threshold = 3000
    stim_onsets = [idx for idx in range(2, len(trigger_channel))
                   if trigger_channel[idx] > threshold and
                      trigger_channel[idx-1] < threshold and
                      trigger_channel[idx-2] < threshold]
    trigg_stim = np.array(stim_onsets) / fs_stim

    # --- Load SEEG trigger events ---
    fs_data = raw_fif.info['sfreq']
    orig_events, _ = mne.events_from_annotations(raw_fif, verbose=False)
    offset_samples = int((offset / 1000) * fs_data)

    # --- Get most frequent event type (main trigger) ---
    event_code = Counter(orig_events[:, 2]).most_common(1)[0][0]
    trigg_data_samples = orig_events[orig_events[:, 2] == event_code, 0].squeeze()
    trigg_data_samples = trigg_data_samples + offset_samples
    trigg_data_secondes = trigg_data_samples / fs_data

    # --- Align triggers with iterative delay correction ---
    threshold_sec = 0.005
    i = 0
    while i < min(len(trigg_data_secondes), len(trigg_stim)) - 5:
        if len(trigg_data_secondes) == len(trigg_stim):
            trigg_stim_zero = trigg_stim - trigg_stim[0]
            trigg_data_zero = trigg_data_secondes - trigg_data_secondes[0]
        elif len(trigg_data_secondes) < len(trigg_stim):
            trigg_stim_zero = trigg_stim - trigg_stim[i]
            trigg_data_zero = trigg_data_secondes - trigg_data_secondes[0]
        else:
            trigg_stim_zero = trigg_stim - trigg_stim[0]
            trigg_data_zero = trigg_data_secondes - trigg_data_secondes[i]

        delay_matrix = np.subtract.outer(trigg_data_zero, trigg_stim_zero)

        if len(trigg_data_zero) <= len(trigg_stim_zero):
            trigg_stim_synced = [trigg_stim_zero[np.argmin(np.abs(delay_matrix[j, :]))]
                                 for j in range(len(trigg_data_zero))]
            paired_diff = trigg_data_zero - np.array(trigg_stim_synced)
        else:
            trigg_data_synced = [trigg_data_zero[np.argmin(np.abs(delay_matrix[:, j]))]
                                 for j in range(len(trigg_stim_zero))]
            paired_diff = np.array(trigg_data_synced) - trigg_stim_zero

        delay_ms = np.abs(round(np.mean(paired_diff), 4)) * 1000
        print(f"Trigger index shift: {i}")
        print(f"Mean delay: {delay_ms:.2f} ms")

        if np.abs(round(np.mean(paired_diff), 4)) > threshold_sec:
            i += 1
            continue
        else:
            # --- Extract aligned data ---
            if i == 0:
                raw_array_ = raw_array[trigg_data_samples[0]:trigg_data_samples[-1], :]
            elif len(trigg_data_secondes) < len(trigg_stim):
                raw_array_ = raw_array[trigg_data_samples[0]:trigg_data_samples[-1], :]
                padding = int((trigg_stim[i] - trigg_stim[0]) * fs_data)
                raw_array_ = np.concatenate((np.zeros((padding, raw_array_.shape[1])), raw_array_), axis=0)
            else:
                raw_array_ = raw_array[trigg_data_samples[i]:trigg_data_samples[-1], :]
            break

    # --- Final alignment quality report ---
    paired_count = min(len(trigg_data_zero), len(trigg_stim_zero))
    std_ms = np.std(paired_diff) * 1000
    print(f"✔️  Aligned {paired_count} trigger pairs.")
    print(f"📉  Lost triggers: {len(trigg_stim) - paired_count}")
    print(f"📏  Std deviation of alignment (ms): {std_ms:.2f}")
    
    # Optional rating of alignment
    if std_ms < 5:
        print("✅ Excellent alignment")
    elif std_ms < 20:
        print("🟡 Acceptable alignment with some jitter")
    else:
        print("🔴 Poor alignment – investigate further")

    return raw_array_
    #%%
