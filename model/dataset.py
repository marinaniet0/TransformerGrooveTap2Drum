import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle
import sys
from tqdm import tqdm
import wandb
import copy
import numpy as np


sys.path.insert(1, "../../hvo_sequence/")
sys.path.insert(1, "../hvo_sequence/")
from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

default_subset_info = {
    "metadata_csv_filename": "metadata.csv",
    "hvo_pickle_filename": "hvo_sequence_data.obj",
    "pickle_source_path": "../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2"
                          "/Processed_On_17_05_2021_at_22_32_hrs",
    "subset_name": "GrooveMIDI_processed_train"
}

default_tap_params = {
    "tapped_sequence_voice": "HH_CLOSED",
    "tapped_sequence_collapsed": False,
    "tapped_sequence_velocity_mode": 1,
    "tapped_sequence_offset_mode": 3
}

default_max_len = 32


def process_dataset(subset, metadata, max_len, tappify_params):
    """Process subset of GMD dataset for Tap2Drum

    @:param subset
        Preprocessed subset of GMD, loaded from https://github.com/behzadhaki/preprocessed_dataset
    @:param metadata: DataFrame
        Pandas DF with the subset information
    @:param max_len: int
        Maximum length for the hvo sequences (by default 32)
    @:param tappify_params: dict
        Dictionary containing the parameters for the flatten_voices function that generates the tapped sequences to be
        used as inputs
            - tapped_sequence_voice
            - tapped_sequence_collapsed
            - tapped_sequence_velocity_mode
            - tapped_sequence_offset_mode
    @:return tuple with inputs (tapped sequences), outputs (full-beats) and hvo_sequences (full hvo objects)
    """
    inputs = []
    outputs = []
    hvo_sequences = []
    tapped_voice_idx = list(ROLAND_REDUCED_MAPPING.keys()).index(tappify_params["tapped_sequence_voice"])
    for idx, hvo_seq in enumerate(tqdm(subset)):
        if len(hvo_seq.time_signatures) == 1:
            all_zeros = not np.any(hvo_seq.hvo.flatten())
            if not all_zeros:
                # Add metadata to hvo_sequence
                hvo_seq.drummer = metadata.loc[idx].at["drummer"]
                hvo_seq.session = metadata.loc[idx].at["session"]
                hvo_seq.master_id = metadata.loc[idx].at["master_id"]
                hvo_seq.style_primary = metadata.loc[idx].at["style_primary"]
                hvo_seq.style_secondary = metadata.loc[idx].at["style_secondary"]
                hvo_seq.beat_type = metadata.loc[idx].at["beat_type"]

                pad_count = max(max_len - hvo_seq.hvo.shape[0], 0)
                hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, pad_count), (0, 0)), "constant")
                hvo_seq.hvo = hvo_seq.hvo[:max_len, :]  # in case seq exceeds max len
                hvo_sequences.append(hvo_seq)
                flat_seq = hvo_seq.flatten_voices(voice_idx=tapped_voice_idx,
                                              reduce_dim=tappify_params["tapped_sequence_collapsed"],
                                              offset_aggregator_modes=tappify_params["tapped_sequence_offset_mode"],
                                              velocity_aggregator_modes=tappify_params["tapped_sequence_velocity_mode"])
                inputs.append(flat_seq)
                outputs.append(hvo_seq.hvo)

    # Load data onto device
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = torch.FloatTensor(inputs).to(dev)
    outputs = torch.FloatTensor(outputs).to(dev)

    return inputs, outputs, hvo_sequences

class GrooveMidiDatasetTap2Drum(Dataset):
    """
    Class that loads and processes the GrooveMidiDataset for the task of Tap2Drum

    Parameters
    ----------
    subset
        Preprocessed subset of GMD, loaded from https://github.com/behzadhaki/preprocessed_dataset
    subset_info: dict
        Dictionary containing the necessary parameters for further processing
            - metadata_csv_filename
            - hvo_pickle_filename
            - pickle_source_path
            - subset_name
    max_len: int
        Maximum length for the hvo sequences (by default 32)
    tappify_params: dict
        Dictionary containing the parameters for the flatten_voices function that generates the tapped sequences to be
        used as inputs
            - tapped_sequence_voice
            - tapped_sequence_collapsed
            - tapped_sequence_velocity_mode
            - tapped_sequence_offset_mode

    Attributes
    ----------
    inputs : list
        list of processed inputs
    outputs : list
        list of outputs
    sequences : str
        list of hvo_sequence objects
    Methods
    -------
    get_hvo_sequence(idx)
        Gets the hvo sequence object given an index
    """

    def __init__(self,
                 subset,
                 subset_info=default_subset_info,
                 **kwargs):

        # Get kwargs
        max_len = kwargs.get("max_len", default_max_len)
        tappify_params = kwargs.get("tappify_params", default_tap_params)

        print("Loading dataset...")

        # Loading metadata onto DataFrame
        metadata = pd.read_csv(os.path.join(subset_info["pickle_source_path"], subset_info["subset"],
                                            subset_info["metadata_csv_filename"]))

        # Get processed inputs, outputs and hvo sequences
        self.inputs, self.outputs, self.sequences = process_dataset(subset, metadata, max_len, tappify_params)

        # wandb.config.update({"set_length": len(self.sequences)})
        print("Dataset loaded\n")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], idx

    def get_hvo_sequence(self, idx):
        return self.sequences[idx]
