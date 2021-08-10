import sys
import pickle
import pandas as pd


sys.path.insert(1, "../../GrooveEvaluator/")
sys.path.insert(1, "../GrooveEvaluator/")
from GrooveEvaluator.evaluator import Evaluator

params = {
    "train_dataset": {
        "pickle_source_path": "../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.5"
                              "/Processed_On_14_06_2021_at_14_26_hrs",
        "subset": "GrooveMIDI_processed_train",
        "metadata_csv_filename": "metadata.csv",
        "hvo_pickle_filename": "hvo_sequence_data.obj",
        "filters": {
            "beat_type": ["beat"],
            "time_signature": ["4-4"]
        },
        "max_len": 32
    },
    "test_dataset": {
        "pickle_source_path": "../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.5"
                              "/Processed_On_14_06_2021_at_14_26_hrs",
        "subset": "GrooveMIDI_processed_test",
        "metadata_csv_filename": "metadata.csv",
        "hvo_pickle_filename": "hvo_sequence_data.obj",
        "filters": {
            "beat_type": ["beat"],
            "time_signature": ["4-4"]
        },
        "max_len": 32
    }
}

if __name__ == "__main__":

    styles = ["hiphop", "funk", "reggae", "soul", "latin", "jazz", "pop", "afrobeat", "highlife", "punk", "rock"]

    list_of_filter_dicts_for_subsets = []
    for style in styles:
        list_of_filter_dicts_for_subsets.append(
           {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
        )


    # TRAIN EVALUATOR
    train_evaluator = Evaluator(
        pickle_source_path=params["train_dataset"]["pickle_source_path"],
        set_subfolder=params["train_dataset"]["subset"],
        hvo_pickle_filename=params["train_dataset"]["hvo_pickle_filename"],
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        max_hvo_shape=(32, 27),
        n_samples_to_use=2048,
        n_samples_to_synthesize_visualize_per_subset=10,
        disable_tqdm=False,
        analyze_heatmap=True,
        analyze_global_features=True,
        _identifier="Train_Set"
    )

    # TEST EVALUATOR
    test_evaluator = Evaluator(
        pickle_source_path=params["test_dataset"]["pickle_source_path"],
        set_subfolder=params["test_dataset"]["subset"],
        hvo_pickle_filename=params["test_dataset"]["hvo_pickle_filename"],
        list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
        max_hvo_shape=(32, 27),
        n_samples_to_use=2048,
        n_samples_to_synthesize_visualize_per_subset=10,
        disable_tqdm=False,
        analyze_heatmap=True,
        analyze_global_features=True,
        _identifier="Test_Set"
    )

    train_evaluator_file_handle = open('train.evaluator', 'wb')
    pickle.dump(train_evaluator, train_evaluator_file_handle)
    train_evaluator_file_handle.close()

    test_evaluator_file_handle = open('test.evaluator', 'wb')
    pickle.dump(test_evaluator, test_evaluator_file_handle)
    test_evaluator_file_handle.close()