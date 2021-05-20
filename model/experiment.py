import data_loader
import torch
import sys

sys.path.append('../../preprocessed_dataset/')

from Subset_Creators.subsetters import GrooveMidiSubsetter

filters = {"beat_type": ["beat"], "time_signature": ["4-4"], "master_id": ["drummer9/session1/9"]}

# LOAD SMALL TRAIN SUBSET
subset_info = {"pickle_source_path": '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.2'
                                     '/Processed_On_17_05_2021_at_22_32_hrs',
               "subset": 'GrooveMIDI_processed_train',
               "metadata_csv_filename": 'metadata.csv',
               "hvo_pickle_filename": 'hvo_sequence_data.obj',
               "filters": filters
               }

if __name__ == "__main__":
    gmd_subsetter = GrooveMidiSubsetter(pickle_source_path=subset_info["pickle_source_path"],
                                        subset=subset_info["subset"],
                                        hvo_pickle_filename=subset_info["hvo_pickle_filename"],
                                        list_of_filter_dicts_for_subsets=[filters])

    _, subset_list = gmd_subsetter.create_subsets()
    gmd = data_loader.GrooveMidiDataset(subset=subset_list[0], subset_info=subset_info)
    inputs, outputs, idx = gmd[0]
    print(inputs)
