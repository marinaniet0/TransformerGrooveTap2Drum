import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
import sys
from tqdm import tqdm
import wandb

sys.path.append('../../hvo_sequence/')
import numpy as np

from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

torch.set_printoptions(profile="full")

subs_info = {
    "metadata_csv_filename": 'metadata.csv',
    "hvo_pickle_filename": 'hvo_sequence_data.obj',
    "pickle_source_path": "../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2"
                          "/Processed_On_17_05_2021_at_22_32_hrs",
    "subset_name": 'GrooveMIDI_processed_train'
}

tap_params = {
    "tapped_sequence_voice": 'HH_CLOSED',
    "tapped_sequence_collapsed": False,
    "tapped_sequence_velocity_mode": 1,
    "tapped_sequence_offset_mode": 3
}


class GrooveMidiDataset(Dataset):
    def __init__(self,
                 subset,
                 subset_info=subs_info,
                 **kwargs):

        metadata = pd.read_csv(os.path.join(subset_info["pickle_source_path"], subset_info["subset"],
                                            subset_info["metadata_csv_filename"]))

        max_len = kwargs.get('max_len', 32)
        tappify_params = kwargs.get('tappify_params', tap_params)

        self.inputs = []
        self.outputs = []
        self.sequences = []

        tapped_voice_idx = list(ROLAND_REDUCED_MAPPING.keys()).index(tappify_params["tapped_sequence_voice"])
        print('Loading dataset...')
        for ix, hvo_seq in enumerate(tqdm(subset)):
            if len(hvo_seq.time_signatures) == 1:
                all_zeros = not np.any(hvo_seq.hvo.flatten())
                if not all_zeros:
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
                    flat_seq = hvo_seq.flatten_voices(voice_idx=tapped_voice_idx,
                                                      reduce_dim=tappify_params[
                                                          "tapped_sequence_collapsed"],
                                                      offset_aggregator_modes=tappify_params[
                                                          "tapped_sequence_offset_mode"],
                                                      velocity_aggregator_modes=tappify_params[
                                                          "tapped_sequence_offset_mode"])
                    self.inputs.append(flat_seq)
                    self.outputs.append(hvo_seq.hvo)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.inputs = torch.FloatTensor(self.inputs).to(dev)
        self.outputs = torch.FloatTensor(self.outputs).to(dev)
        wandb.config.update({"set_length": len(self.sequences)})
        print('Dataset loaded\n')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], idx

    def get_hvo_sequence(self, idx):
        return self.sequences[idx]
