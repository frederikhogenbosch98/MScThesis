'''
Based on the work in ECGNeuralNetwork repository by lorenzobrusco.
Original repository: https://github.com/lorenzobrusco/ECGNeuralNetwork
'''

import json
import matplotlib.pyplot as plt
import tqdm
import random
import cv2
from os.path import isfile, join
from os import listdir
import os
import wfdb
import numpy as np
from scipy.signal import resample
from PIL import Image
import io

_range_to_ignore = 30
_directory = '../physionet/incart/' # incart files location
_dataset_dir = 'data/incart/' # output folder



def create_img_from_sign(lblabels, lbrevert_labels, lboriginal_labels, size=(224, 224), augmentation=True):

    if not os.path.exists(_directory):
        os.makedirs(_directory)

    files = [f[:-4] for f in listdir(_directory) if isfile(join(_directory, f)) if (f.find('.dat') != -1)]

    for file in files:
        sig, _ = wfdb.rdsamp(_directory + file)
        ann = wfdb.rdann(_directory + file, extension='atr')
        len_sample = len(ann.sample)
        for i in tqdm.tqdm(range(1, len_sample - 2)):
            if ann.symbol[i] not in lboriginal_labels:
                continue
            label = lboriginal_labels[ann.symbol[i]]
            dir = '{}/{}'.format(_dataset_dir, label)
            if not os.path.exists(dir):
                os.makedirs(dir)

            start = ann.sample[i - 1] + _range_to_ignore
            end = ann.sample[i + 1] - _range_to_ignore
            
            plot_x = [sig[i][0] for i in range(start, end)]
            plot_y = [i * 1 for i in range(start, end)]

            fig = plt.figure(frameon=False)
            plt.plot(plot_y, plot_x)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            filename = f'{_dataset_dir}/{label}/{label}_{file[-3:]}{start}{end}.png' 

            buf = create_img(plot_x, 224, 224)
            image_pil = Image.open(buf)
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
            cv2.imwrite(filename, image_cv)
            plt.cla()
            plt.clf()
            plt.close('all')


def create_img(signal, width, height):

    dpi = 230 
    fig_width_in = width / dpi
    fig_height_in = height / dpi
    t = np.linspace(0, 1, len(signal))  

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')


    ax.plot(t, signal, color='black', linewidth=0.5)
    ax.axis('off')
    buf = io.BytesIO()

    plt.savefig(buf, dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close(fig)
    
    return buf


if __name__ == "__main__":
    labels_json = '{ ".": "N", "N": "N", "V": "V", "/": "Q", "L": "N", "R": "N", "A": "S", "a": "S", "J": "S", "S":"S", "F":"F", "e":"N", "j":"N", "E":"V", "f":"Q", "Q":"Q"}'
    labels_to_float = '{ "N": "0", "S" : "1", "V": "2", "F": "3", "Q": "4"}'
    float_to_labels = '{ "0": "N", "1" : "S", "2": "V", "3": "F", "4": "Q"}'
    labels = json.loads(labels_to_float)
    revert_labels = json.loads(float_to_labels)
    original_labels = json.loads(labels_json)

    create_img_from_sign(lblabels=labels, lbrevert_labels=revert_labels, lboriginal_labels=original_labels)
