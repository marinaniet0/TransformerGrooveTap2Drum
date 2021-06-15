import sys
import unittest
import torch
from model.dataset import GrooveMidiDatasetTap2Drum

sys.path.insert(1, "../../preprocessed_dataset/")
sys.path.insert(1, "../preprocessed_dataset/")
from Subset_Creators.subsetters import GrooveMidiSubsetter

params = {
        "dataset": {
            "pickle_source_path": "../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.4"
                                  "/Processed_On_09_06_2021_at_12_41_hrs",
            "subset": "GrooveMIDI_processed_train",
            "metadata_csv_filename": "metadata.csv",
            "hvo_pickle_filename": "hvo_sequence_data.obj",
            "filters": {
                "beat_type": ["beat"],
                "time_signature": ["4-4"]
                # "master_id": ["drummer9/session1/8"]
            },
            "max_len": 32
        },
        "tappify_params": {
            "tapped_sequence_voice": "HH_CLOSED",
            "tapped_sequence_collapsed": False,
            "tapped_sequence_velocity_mode": 1,
            "tapped_sequence_offset_mode": 3
        }
    }


class Test_Tap2DrumDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, subset_list = GrooveMidiSubsetter(pickle_source_path=params["dataset"]["pickle_source_path"],
                                             subset=params["dataset"]["subset"],
                                             hvo_pickle_filename=params["dataset"]["hvo_pickle_filename"],
                                             list_of_filter_dicts_for_subsets=[
                                                 params["dataset"]["filters"]]).create_subsets()

        cls.gmd = GrooveMidiDatasetTap2Drum(subset=subset_list[0], subset_info=params["dataset"],
                                        tappify_params=params["tappify_params"], max_len=params["dataset"]["max_len"])


    def test_tensor_shape(self):
        for (input, output, idx) in Test_Tap2DrumDataset.gmd:
            self.assertEqual(list(input.shape), [32, 27])
            self.assertEqual(list(output.shape), [32, 27])

    def test_no_zero_tensors(self):
        for (input, output, idx) in Test_Tap2DrumDataset.gmd:
            self.assertGreater(torch.count_nonzero(input), 0)
            self.assertGreater(torch.count_nonzero(output), 0)

