import argparse, sys, os, re, pickle
import pandas as pd
import numpy as np

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from helper_functions import parse_input_path, parse_output_path, get_FRET_distance

ch_names_dict = dict(
    ch001='1_2_3',
    ch002='2_3_4',
    ch003='1_2_4',
    ch004='1_3_4',
    ch005='1_2_3_4',
    ch007='1_2_3',
    ch008='2_3_4',
    ch009='1_2_4',
    ch010='1_3_4',
    ch011='1_2_3_4',
    ch015='2_3_4_p3',
    ch016='2_3_4_p1',
    ch017='2_3_4_p2',
    ch018='2_3_4_p0',
    ch01001='2_3_4_p3',
    ch01002='2_3_4_p1',
    ch01003='2_3_4_p2',
    ch01004='2_3_4_p0',
    ch01005='2_3_4_original_vs_p3',
    ch01006='2_3_4_original_vs_p1',
    ch01007='2_3_4_original_vs_p2',
    ch01008='2_3_4_p1_vs_p3',
    ch01009='2_3_4_p2_vs_p3',
    ch01010='2_3_4_p1_vs_p2'
)

class ImaxExperimentData(object):
    def __init__(self, fret_resolution, txt_fn, exp_id, donor_only_threshold):
        self.data_dict = {'X': {'nb_tags': 0, 'fret': [], 'dist': [], 'fingerprint': [], 'coords': []}}

        self.exp_id = exp_id
        self.txt_fn = txt_fn
        self.fret_resolution = fret_resolution
        self.donor_only_threshold = donor_only_threshold

        self.parse_txt_file()

    @property
    def nb_molecules(self):
        return len(self.data_dict['X']['fret'])


    @property
    def fret_resolution(self):
        return self._fret_resolution

    @fret_resolution.setter
    def fret_resolution(self, fr):
        self._fret_resolution = fr
        self._res_bins = np.arange(0.01, 1 + fr, fr)

    def parse_txt_file(self):

        with open(self.txt_fn, 'r') as fh:
            txt_str = fh.read()
        for mol_txt in txt_str.split('\n\n')[1:]:
            self.add_mol(mol_txt)

    def add_mol(self, mt):
        mt_list = mt.split('\n')
        mol_idx = int(re.search('(?<=Molecules )[0-9]+', mt_list.pop(0)).group(0))
        event_dict = {}
        for x in mt_list:
            x = x.strip()
            if not len(x): continue
            if 'No barcode found' in x: return
            if x.startswith('Barcode'):
                barcode_idx = int(x.split(' ')[1][:-1])
                event_dict[barcode_idx] = {}
            else:
                var_name, value = x.rsplit(' ', 1)
                event_dict[barcode_idx][var_name] = float(value)

        fret_fp = [event_dict[ed]['position'] for ed in event_dict]
        fret_fp = [x for x in fret_fp if x > self.donor_only_threshold]  # filter out donor-only
        dist_fp = [get_FRET_distance(ef) for ef in fret_fp]
        hist_fp = np.clip((np.histogram(fret_fp, bins=self._res_bins)[0] > 0).astype(float), 0, 1)
        nbv = len(fret_fp)
        nb_tags = int(np.ceil((1 + np.sqrt(8 * nbv)) / 2))  # estimate of number tags. If not round, must be because peaks are obscured, so ceil

        self.data_dict['X']['fret'].append(fret_fp)
        self.data_dict['X']['dist'].append(dist_fp)
        self.data_dict['X']['fingerprint'].append(hist_fp)
        self.data_dict['X']['nb_tags'] = max(self.data_dict['X']['nb_tags'], nb_tags)

    def save(self, out_dir):
        fret_dict = {
            'up_id': self.exp_id,
            'anchor_type': 'other',
            'efficiency': 1.0,
            'nb_examples': self.nb_molecules,
            'data': self.data_dict
        }
        with open(f'{out_dir}{self.exp_id}.pkl', 'wb') as fh:
            pickle.dump(fret_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)


def main(txt_list, fret_resolution, donor_only_threshold, out_dir):
    res_bins = np.arange(0.01, 1 + fret_resolution, fret_resolution)
    out_dir = parse_output_path(out_dir)

    for txt_fn in txt_list:
        exp_id = ch_names_dict[re.search('ch[0-9]+', Path(txt_fn).stem).group(0)]
        exp_data = ImaxExperimentData(fret_resolution, txt_fn, exp_id, donor_only_threshold)
        exp_data.save(out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse text files into table format')
    parser.add_argument('--in-dir', type=str, required=True)
    parser.add_argument('--donor-only-threshold', type=float, default=0.05)
    parser.add_argument('--fret-resolution', type=float, default=0.01)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()

    txt_list = parse_input_path(args.in_dir, pattern='*.txt')
    main(txt_list, args.fret_resolution, args.donor_only_threshold, args.out_dir)
