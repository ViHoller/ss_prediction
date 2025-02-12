import numpy as np
import keras.backend     as backend

all_aa = "ARNDCEQGHILKMFPSTWYVX"
aa_onehot_dict = dict()
for i, aa in enumerate(all_aa):
    aa_onehot_dict[aa] = i
ss_map = {'C': 0, 'H': 1, 'E': 2}


def aa_onehot_encoding(seq, padding=True):
    profile = []
    for aa in seq:
        encoded = np.zeros(21)
        encoded[aa_onehot_dict[aa]] = 1
        profile.append(encoded)
    if padding:
        while len(profile) != 800:  # pad to 800
            profile.append(np.zeros(21))
    return profile


def parse_dssp(path, dssp_file):
    with open(path+"dssp/"+dssp_file+".dssp", 'r') as file:
        file.readline()
        ss = file.readline().rstrip()
    return ss


def parse_pssm(path, pssm_filename, padding=True):
    profile = []
    seq = ''
    with open(path+"pssm/"+pssm_filename+".pssm", 'r') as pssm:
        pssm_lines = pssm.readlines()
        for line in pssm_lines[3:-6]:
            line = line.rstrip().split()
            seq += line[1]
            profile_line = []
            for n in line[22:-2]:
                profile_line.append(float(n)/100)
            profile.append(profile_line)
    if padding:
        while (len(profile) != 800):
            profile.append(np.zeros(20))
    return profile, seq


def ss_onehot_encoding(ss_sequence, padding=True):
    ss_encoded = []
    for struc in ss_sequence:
        encoding = np.zeros(3)
        encoding[ss_map[struc]] = 1
        ss_encoded.append(encoding)
    if padding:
        while (len(ss_encoded) != 800):
            ss_encoded.append(np.zeros(3))
    return ss_encoded


def get_data(path, file, encode_y=True, padding_x=True, padding_y=True, test=False):
    x = []
    y = []
    with open(path+file, 'r') as sample_file:  # add some stuff to check?
        for line in sample_file:
            line = line.rstrip()
            if test:
                line = line.replace(':', '_')
            pssm, sequence = parse_pssm(path, line, padding=padding_x)
            sequence_hot = aa_onehot_encoding(sequence, padding=padding_x)
            features = np.concatenate((sequence_hot, pssm), axis=1)
            x.append(features)

            dssp = parse_dssp(path, line).replace('-', 'C')
            if encode_y:
                dssp = ss_onehot_encoding(dssp, padding=padding_y )

            y.append(dssp)
    return x, y


def truncated_accuracy(y_true, y_pred):
    mask = backend.sum(y_true, axis=2)
    y_pred_labels = backend.cast(backend.argmax(y_pred, axis=2), 'float32')
    y_true_labels = backend.cast(backend.argmax(y_true, axis=2), 'float32')
    is_same = backend.cast(backend.equal(
        y_true_labels, y_pred_labels), 'float32')
    num_same = backend.sum(is_same * mask, axis=1)
    lengths = backend.sum(mask, axis=1)
    return backend.mean(num_same / lengths, axis=0)
