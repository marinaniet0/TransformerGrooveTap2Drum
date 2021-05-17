import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
import sys
sys.path.append('../../hvo_sequence/')
import numpy as np

from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

filters = {
    "drummer": None,
    "session": None,
    "loop_id": None,
    "master_id": None,
    "style_primary": None,
    "bpm": None,
    "beat_type": ["beat"],
    "time_signature": ["4-4"],
    "full_midi_filename": None,
    "full_audio_filename": None
}


def check_if_passes_filters(obj, filters):
    for key in filters:
        if filters[key] is not None and obj.to_dict()[key] not in filters[key]:
            return False
    return True


class GrooveMidiDataset(Dataset):
    def __init__(self,
                 source_path='../../preprocessed_dataset/datasets_zipped/GrooveMidi/hvo_0.3.0/'
                             'Processed_On_13_05_2021_at_12_56_hrs',
                 subset='train',
                 metadata_csv_filename='metadata.csv',
                 hvo_pickle_filename='hvo_sequence_data.obj',
                 filters=filters,
                 max_len=32,
                 device_str='cpu',
                 tapped_sequence_voice='HH_CLOSED',
                 tapped_sequence_collapsed=False,
                 tapped_sequence_velocity_mode=1,
                 tapped_sequence_offset_mode=3):

        subset_str = 'GrooveMIDI_processed_' + subset

        train_file = open(os.path.join(source_path, subset_str, hvo_pickle_filename), 'rb')
        train_set = pickle.load(train_file)
        metadata = pd.read_csv(os.path.join(source_path, subset_str, metadata_csv_filename))

        self.inputs = []
        self.outputs = []
        self.sequences = []

        tapped_voice_idx = list(ROLAND_REDUCED_MAPPING.keys()).index(tapped_sequence_voice)

        for ix, hvo_seq in enumerate(train_set):
            if len(hvo_seq.time_signatures) == 1:
                all_zeros = not np.any(hvo_seq.hvo.flatten())
                if not all_zeros:
                    if check_if_passes_filters(metadata.loc[ix], filters):
                        hvo_seq.drummer = metadata.loc[ix].at["drummer"]
                        hvo_seq.session = metadata.loc[ix].at["session"]
                        hvo_seq.master_id = metadata.loc[ix].at["master_id"]
                        hvo_seq.style_primary = metadata.loc[ix].at["style_primary"]
                        hvo_seq.style_secondary = metadata.loc[ix].at["style_secondary"]
                        hvo_seq.beat_type = metadata.loc[ix].at["beat_type"]

                        pad_count = max(max_len - hvo_seq.hvo.shape[0], 0)
                        hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, pad_count), (0, 0)), 'constant')
                        hvo_seq.hvo = hvo_seq.hvo[:max_len, :]  # in case seq exceeds max len
                        self.sequences.append(hvo_seq)
                        self.inputs.append(hvo_seq.flatten_voices(voice_idx=tapped_voice_idx,
                                                                  reduce_dim=tapped_sequence_collapsed))
                        self.outputs.append(hvo_seq.hvo)

        dev = torch.device(device_str)
        self.inputs = torch.Tensor(self.inputs, device=dev).to(torch.float)
        self.outputs = torch.Tensor(self.outputs, device=dev).to(torch.float)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], idx

    def get_hvo_sequence(self, idx):
        return self.sequences[idx]

