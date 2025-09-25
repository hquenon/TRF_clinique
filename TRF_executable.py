# ==== Imports ====
import os
import sys
import copy
import math
import matplotlib.pyplot as plt
import mne
import numpy as np
import pickle
from collections import Counter
from pydub import AudioSegment
import re
from scipy import stats
import scipy.io
from sklearn.model_selection import KFold
from spyeeg.models.TRF import TRFEstimator
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess

import funcs

def launch_matlab_script(matlab_script_path, args=''):
    command = f'matlab -batch \"run(\'{matlab_script_path}\') {args}\"'
    subprocess.run(command, shell=True)

# === Sélection des fichiers et dossiers ===

# Sélectionner le(s) fichier(s) sorcière .vhdr
sorciere_path = filedialog.askopenfilenames(
    title="Sélectionner le fichier Sorcière (.vhdr)",
    filetypes=[("EEG BrainVision files", "*.vhdr")]
)
if not sorciere_path:
    messagebox.showerror("Erreur", "Aucun fichier Sorcière sélectionné.")
    sys.exit()

# Sélectionner le(s) fichier(s) pianos .vhdr
piano_path = filedialog.askopenfilenames(
    title="Sélectionner les fichiers Pianos (.vhdr)",
    filetypes=[("EEG BrainVision files", "*.vhdr")]
)
if not piano_path:
    messagebox.showerror("Erreur", "Aucun fichier Pianos sélectionné.")
    sys.exit()

# Sélectionner le dossier de sortie
out_path = filedialog.askdirectory(
    title="Sélectionner le dossier de travail (sauvegarde résultats)"
)
if not out_path:
    messagebox.showerror("Erreur", "Aucun dossier de sortie sélectionné.")
    sys.exit()


# Impression pour vérification
print("Sorcière fichiers sélectionnés:")
for path in sorciere_path:
    print(path)
print("\nPianos fichiers sélectionnés:")
for path in piano_path:
    print(path)
print("\nDossier de sortie:")
print(out_path)


# ==== Chargement des fichiers EEG ====
sorciere_mne = funcs.vhdr_to_mne(sorciere_path)
pianos_mne = funcs.vhdr_to_mne(piano_path)
piano_folder = os.path.dirname(piano_path[0])
sorciere_folder = os.path.dirname(sorciere_path[0])

# Impression pour vérification
print("\nDossier Sorcière:", sorciere_folder)
print("Dossier Pianos:", piano_folder)

# ==== Lister les fichiers audio (stimulus) ====
stim_piano = [
    file for file in os.listdir(os.path.join(piano_folder, 'stim'))
    if os.path.isfile(os.path.join(piano_folder, 'stim', file)) and file.endswith('.mp3')
]
stim_sorciere = [
    file for file in os.listdir(os.path.join(sorciere_folder, 'stim'))
    if os.path.isfile(os.path.join(sorciere_folder, 'stim', file)) and file.endswith('.mp3')
]
print("\nFichiers stimulus Pianos:", stim_piano)
print("Fichiers stimulus Sorcière:", stim_sorciere)

# ==== Montage bipolaire ====

# Pour Sorcière
sorciere_bipol, channels_bipol = funcs.bipolar_montage(sorciere_mne[0], chans=True)
sorciere_bipol = [sorciere_bipol]  # mise en liste pour homogénéité

# Pour Pianos
pianos_bipol = []
for piano in pianos_mne:
    piano_bipol = funcs.bipolar_montage(piano, chans=False)
    pianos_bipol.append(piano_bipol)

# ==== Tri des channels ====

def custom_sort(value):
    alpha_part = ''.join(filter(lambda x: x.isalpha() or x == "'", value))
    num_part = int(''.join(filter(str.isdigit, value)))
    return alpha_part, num_part

sorted_channels_bipol = sorted(channels_bipol, key=custom_sort)

# Réorganiser Sorcière selon tri
sorciere_bipol_sorted = [np.empty_like(sorciere_bipol[0])]
for i, channel_name in enumerate(sorted_channels_bipol):
    channel_index = channels_bipol.index(channel_name)
    sorciere_bipol_sorted[0][:, i] = sorciere_bipol[0][:, channel_index]

# Réorganiser Pianos selon tri
pianos_bipol_sorted = []
for piano in pianos_bipol:
    sorted_piano_bipol = np.empty_like(piano)
    for i, channel_name in enumerate(sorted_channels_bipol):
        channel_index = channels_bipol.index(channel_name)
        sorted_piano_bipol[:, i] = piano[:, channel_index]
    pianos_bipol_sorted.append(sorted_piano_bipol)

# ==== Sauvegarde des channels pour Matlab ====
scipy.io.savemat(os.path.join(out_path, 'channels.mat'), {'ch': np.array(sorted_channels_bipol)})

# ==== Alignement EEG <> Audio ====

# Sorcière
aligned_sorciere = funcs.align(
    raw_fif=sorciere_mne[0],
    raw_array=sorciere_bipol_sorted[0],
    stim_dir=os.path.join(sorciere_folder, 'stim', stim_sorciere[0]),
    offset=0
)

# Pianos
aligned_pianos = []
for i in range(len(pianos_bipol_sorted)):
    aligned = funcs.align(
        raw_fif=pianos_mne[i],
        raw_array=pianos_bipol_sorted[i],
        stim_dir=os.path.join(piano_folder, 'stim', stim_piano[i]),
        offset=0
    )
    aligned_pianos.append(aligned)

print("\nAlignement terminé pour Sorcière et Pianos.")

# ==== Préparation EEG : filtrage et downsampling ====

# Paramètres de filtrage
high_pass = 0.5  # Hz
low_pass = 30    # Hz

# Fréquence d'acquisition EEG
fs_sorciere = sorciere_mne[0].info['sfreq']

# Sorcière
prepared_sorciere = funcs.prepare_data(
    data=aligned_sorciere,
    picks=list(range(len(sorted_channels_bipol))),  # Tous les channels triés
    high_pass=high_pass,
    low_pass=low_pass,
    fs_data=fs_sorciere
)

# Pianos
prepared_pianos = []
for i, aligned_piano in enumerate(aligned_pianos):
    fs_piano = pianos_mne[i].info['sfreq']
    prepared = funcs.prepare_data(
        data=aligned_piano,
        picks=list(range(len(sorted_channels_bipol))),  # Même tri
        high_pass=high_pass,
        low_pass=low_pass,
        fs_data=fs_piano
    )
    prepared_pianos.append(prepared)

print("\nFiltrage et downsampling terminé pour Sorcière et Pianos.")

# ==== Chargement et assemblage des régressseurs pour Pianos ====

pianos_ultimate, pianos_regs_all, pianos_regs_all_labels, fs = funcs.concatenation(
    pianos=prepared_pianos,
    data_path=piano_folder,
    stim_keeped=stim_piano
)

# ==== Import spécifique pour textgrids si nécessaire ====
import sys
sys.path.append(r"C:\\Users\\nadege\\mambaforge\\Lib\\site-packages")  # Adapter chemin si besoin
import textgrids
print("textgrids importé avec succès.")

# ==== Chargement des régressseurs pour Sorcière ====

sorciere_ultimate, sorciere_regs_all, sorciere_regs_all_label = funcs.load_sorciere_regressors(
    data_path=sorciere_folder,
    data=prepared_sorciere
)

# ==== Sélection des régressseurs pour Sorcière ====

# Acoustic regressors
X_sorc_full = np.squeeze(sorciere_regs_all)
label_to_index = {label: idx for idx, label in enumerate(sorciere_regs_all_label)}

acoustic_names = [
    'loudness', 'brightness', 'periodicity',
    'half_rectified_der_loudness', 'half_rectified_der_brightness', 'half_rectified_der_periodicity',
    'syllabe_onset'
]
predictive_names = ['onsets', 'entropy','surprise']

idx_acoustic = [label_to_index[n] for n in acoustic_names]
idx_predictive = [label_to_index[n] for n in predictive_names]

X_sorc_acoustic = X_sorc_full[:, idx_acoustic]
X_sorc_predictive = X_sorc_full[:, idx_predictive]

# ==== Sélection des régressseurs pour Pianos ====

X_piano_full = np.squeeze(pianos_regs_all)
label_to_index = {label: idx for idx, label in enumerate(pianos_regs_all_labels)}

acoustic_names = ['envelope', 'der_envelope_hr']
predictive_names = ['onset', 'surprise', 'surprise_positive']

idx_acoustic = [label_to_index[n] for n in acoustic_names]
idx_predictive = [label_to_index[n] for n in predictive_names]

X_piano_acoustic = X_piano_full[:, idx_acoustic]
X_piano_predictive = X_piano_full[:, idx_predictive]

# ==== Définir paramètres TRF ====

import gc  # pour garbage collector après sauvegardes lourdes

n_folds = 5
tw_trf = [-2, 2]  # Time window en secondes
alphas = [1e-4, 1e-2, 1e2, 1e4, 1e6, 1e8]

coefs_times = np.linspace(tw_trf[0], tw_trf[1], int(np.diff(tw_trf)[0] * fs + 1))  # Axe temporel des kernels TRF

# ==== Import spyeeg ====
import sys
sys.path.append(r"C:\\Users\\nadege\\mambaforge\\Lib\\site-packages")  # Adapter si besoin
import spyeeg
print("Bibliothèque spyeeg importée avec succès.")

# ==== TRF Sorcière - Acoustic ====
print("\n🧪 Lancement TRF Sorcière - Acoustic...")
sorciere_regs_acoustic_scores_cv, sorciere_regs_acoustic_coefs_fit = funcs.trf_spyeeg(
    regs=np.squeeze(sorciere_regs_acoustic),
    data=np.squeeze(sorciere_ultimate),
    tmin=tw_trf[0],
    tmax=tw_trf[1],
    fs=fs,
    alpha=alphas,
    channels=sorted_channels_bipol
)

scipy.io.savemat(os.path.join(out_path, 'sorciere_acoustic.mat'), {
    'data': sorciere_ultimate,
    'channels': sorted_channels_bipol,
    'r2': sorciere_regs_acoustic_scores_cv,
    'kernels': sorciere_regs_acoustic_coefs_fit,
    'fs': fs
})
print("✅ Terminé.")
gc.collect()

# ==== TRF Sorcière - Predictive ====
print("\n🧪 Lancement TRF Sorcière - Predictive...")
sorciere_regs_predictive_scores_cv, sorciere_regs_predictive_coefs_fit = funcs.trf_spyeeg(
    regs=np.squeeze(sorciere_regs_predictive),
    data=np.squeeze(sorciere_ultimate),
    tmin=tw_trf[0],
    tmax=tw_trf[1],
    fs=fs,
    alpha=alphas,
    channels=sorted_channels_bipol
)

scipy.io.savemat(os.path.join(out_path, 'sorciere_predictive.mat'), {
    'data': sorciere_ultimate,
    'channels': sorted_channels_bipol,
    'r2': sorciere_regs_predictive_scores_cv,
    'kernels': sorciere_regs_predictive_coefs_fit,
    'fs': fs
})
print("✅ Terminé.")
gc.collect()

# ==== TRF Pianos - Acoustic ====
print("\n🧪 Lancement TRF Piano - Acoustic...")
pianos_regs_acoustic_scores_cv, pianos_regs_acoustic_coefs_fit = funcs.trf_spyeeg(
    regs=np.squeeze(pianos_regs_acoustic),
    data=np.squeeze(pianos_ultimate),
    tmin=tw_trf[0],
    tmax=tw_trf[1],
    fs=fs,
    alpha=alphas,
    channels=sorted_channels_bipol
)

scipy.io.savemat(os.path.join(out_path, 'pianos_acoustic.mat'), {
    'data': pianos_ultimate,
    'channels': sorted_channels_bipol,
    'r2': pianos_regs_acoustic_scores_cv,
    'kernels': pianos_regs_acoustic_coefs_fit,
    'fs': fs
})
print("✅ Terminé.")
gc.collect()

# ==== TRF Pianos - Predictive ====
print("\n🧪 Lancement TRF Pianos - Predictive...")
pianos_regs_predictive_scores_cv, pianos_regs_predictive_coefs_fit = funcs.trf_spyeeg(
    regs=np.squeeze(pianos_regs_predictive),
    data=np.squeeze(pianos_ultimate),
    tmin=tw_trf[0],
    tmax=tw_trf[1],
    fs=fs,
    alpha=alphas,
    channels=sorted_channels_bipol
)

scipy.io.savemat(os.path.join(out_path, 'pianos_predictive.mat'), {
    'data': pianos_ultimate,
    'channels': sorted_channels_bipol,
    'r2': pianos_regs_predictive_scores_cv,
    'kernels': pianos_regs_predictive_coefs_fit,
    'fs': fs
})
print("✅ Terminé.")
gc.collect()

# ==== Sauvegarder les scores R² dans un fichier texte ====

# Création du chemin complet pour scores.txt
scores_txt_path = os.path.join(out_path, 'scores.txt')

with open(scores_txt_path, 'w') as file:
    # Écrire l'en-tête avec colonnes alignées
    file.write(f'{"Channel":<15}{"Speech Acoustic (in %)":<25}{"Speech Predictive (in %)":<25}{"Music Acoustic (in %)":<25}{"Music Predictive (in %)":<25}\n')

    # Écrire les données
    for chan, scr1, scr2, scr3, scr4 in zip(
        sorted_channels_bipol,
        sorciere_regs_acoustic_scores_cv,
        sorciere_regs_predictive_scores_cv,
        pianos_regs_acoustic_scores_cv,
        pianos_regs_predictive_scores_cv
    ):
        file.write(f'{chan:<15}{round(scr1*100,2):<25}{round(scr2*100,2):<25}{round(scr3*100,2):<25}{round(scr4*100,2):<25}\n')

print("\nFichier scores.txt généré avec succès.")

# ==== Lancement automatique du script MATLAB pour visualiser les résultats ====

import matlab.engine

# Démarrer le moteur MATLAB
eng = matlab.engine.start_matlab()

# Définir la variable 'out_path' dans MATLAB et exécuter le script
# Chemin du script MATLAB à adapter selon ton poste final
# Sélectionner le script MATLAB .m à exécuter
root = tk.Tk()
root.withdraw()
root.lift()
root.attributes('-topmost', True)  # Force l'apparition au-dessus
matlab_script = filedialog.askopenfilename(
    title="Sélectionner le script MATLAB (.m)",
    filetypes=[("MATLAB files", "*.m")]
)
root.destroy()
if not matlab_script:
    messagebox.showerror("Erreur", "Aucun script MATLAB sélectionné.")
    sys.exit()

print("\nLancement du script MATLAB...")
eng.workspace['out_path'] = out_path  # Passe out_path dans l'espace de travail MATLAB
import io
matlab_stdout = io.StringIO()
eng.run(matlab_script, nargout=0, stdout=matlab_stdout)


# Fermer le moteur MATLAB
eng.quit()

print("\nScript MATLAB exécuté avec succès.")

messagebox.showinfo("Terminé", "Analyse complète réalisée avec succès !\nVous pouvez consulter les résultats.")
