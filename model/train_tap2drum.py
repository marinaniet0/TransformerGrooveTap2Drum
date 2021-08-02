import sys
import wandb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import GrooveMidiDatasetTap2Drum, process_dataset
from utils import eval_log_freq
import pickle

sys.path.insert(1, "../../BaseGrooveTransformers/")
sys.path.insert(1, "../BaseGrooveTransformers/")
from models.train import *

sys.path.insert(1, "../../GrooveEvaluator/")
sys.path.insert(1, "../GrooveEvaluator/")
from GrooveEvaluator.evaluator import Evaluator

sys.path.insert(1, "../../preprocessed_dataset/")
sys.path.insert(1, "../preprocessed_dataset/")
from Subset_Creators.subsetters import GrooveMidiSubsetter

#import os
#os.environ["WANDB_MODE"]="offline"
sys.path.insert(1, "../../hvo_sequence/")
sys.path.insert(1, "../hvo_sequence/")
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

if __name__ == "__main__":

    hyperparameter_defaults = dict(
        optimizer_algorithm="sgd",
        d_model=128,
        n_heads=8,
        dropout=0.1,
        num_encoder_decoder_layers=1,
        learning_rate=1e-3,
        batch_size=64,
        dim_feedforward=512,  # multiple of d_model
        epochs=1,
        loss_hit_penalty_multiplier=1,
        train_eval=1,
        test_eval=1,
        load_evaluator=1
    )

    wandb_run = wandb.init(config=hyperparameter_defaults, project="transformer_groove_tap2drum")

    params = {
        "model": {
            "optimizer": wandb.config.optimizer_algorithm,
            "d_model": wandb.config.d_model,
            "n_heads": wandb.config.n_heads,
            "dim_feedforward": wandb.config.dim_feedforward,
            "dropout": wandb.config.dropout,
            "num_encoder_layers": wandb.config.num_encoder_decoder_layers,
            "num_decoder_layers": wandb.config.num_encoder_decoder_layers,
            "max_len": 32,
            "embedding_size_src": 27,
            "embedding_size_tgt": 27,
            "encoder_only": True,  # Set to false for encoder-decoder
            "loss_hit_penalty_multiplier": wandb.config.loss_hit_penalty_multiplier,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "training": {
            "learning_rate": wandb.config.learning_rate,
            "batch_size": wandb.config.batch_size
        },
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
        },
        "tappify_params": {
            "tapped_sequence_voice": "HH_CLOSED",
            "tapped_sequence_collapsed": False,
            "tapped_sequence_velocity_mode": 1,
            "tapped_sequence_offset_mode": 3
        },
        "load_evaluator": wandb.config.load_evaluator,
        "load_model": None,  # if we don't want to load any model, set to None
        #"load_model": {
        #    "location": "local",
        #    "dir": "./wandb/run-20210609_162149-1tsi1g1n/files/saved_models/",
        #    "file_pattern": "transformer_run_{}_Epoch_{}.Model"
        #}
        #"load_model": {
        #    "location": "wandb",
        #    "dir": "marinaniet0/tap2drum/1tsi1g1n/",
        #    "file_pattern": "saved_models/transformer_run_{}_Epoch_{}.Model",
        #    "epoch": 51,
        #    "run": "1tsi1g1n"
        #}
        "train_eval": wandb.config.train_eval,
        "test_eval": wandb.config.test_eval
    }

    # PYTORCH LOSS FUNCTIONS
    BCE_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    MSE_fn = torch.nn.MSELoss(reduction='none')

    model, optimizer, ep = initialize_model(params)
    wandb.watch(model)

    # DATASET LOADING FOR TRAINING
    _, subset_list = GrooveMidiSubsetter(pickle_source_path=params["train_dataset"]["pickle_source_path"],
                                         subset=params["train_dataset"]["subset"],
                                         hvo_pickle_filename=params["train_dataset"]["hvo_pickle_filename"],
                                         list_of_filter_dicts_for_subsets=[params["train_dataset"]["filters"]]).create_subsets()

    gmd = GrooveMidiDatasetTap2Drum(subset=subset_list[0], subset_info=params["train_dataset"],
                                    tappify_params=params["tappify_params"], max_len=params["train_dataset"]["max_len"])

    dataloader = DataLoader(gmd, batch_size=params["training"]["batch_size"], shuffle=True)

    # Get number of epochs from wandb config
    eps = wandb.config.epochs

    if params["train_eval"] or params["test_eval"]:

        styles = ["hiphop", "funk", "reggae", "soul", "latin", "jazz", "pop", "afrobeat", "highlife", "punk", "rock"]

        list_of_filter_dicts_for_subsets = []
        for style in styles:
            list_of_filter_dicts_for_subsets.append(
               {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
            )

        if params["train_eval"]:

            if params["load_evaluator"]:
                train_evaluator = pickle.load(open('train.evaluator', 'rb'))
            else:
                # TRAIN EVALUATOR
                train_evaluator = Evaluator(
                    pickle_source_path=params["train_dataset"]["pickle_source_path"],
                    set_subfolder=params["train_dataset"]["subset"],
                    hvo_pickle_filename=params["train_dataset"]["hvo_pickle_filename"],
                    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
                    max_hvo_shape=(32, 27),
                    n_samples_to_use=11,
                    n_samples_to_synthesize_visualize_per_subset=10,
                    disable_tqdm=False,
                    analyze_heatmap=True,
                    analyze_global_features=True,
                    _identifier="Train_Set"
                )
            train_evaluator_subset = train_evaluator.get_ground_truth_hvo_sequences()
            metadata_train = pd.read_csv(os.path.join(params["train_dataset"]["pickle_source_path"],
                                                      params["train_dataset"]["subset"],
                                                      params["train_dataset"]["metadata_csv_filename"]))
            print("Generating inputs for train evaluator...")
            train_eval_inputs, _, _ = process_dataset(train_evaluator_subset, metadata=metadata_train,
                                                         max_len=params["train_dataset"]["max_len"],
                                                         tappify_params=params["tappify_params"])
            print("Inputs for train evaluator generated.")

        if params["test_eval"]:
            if params["load_evaluator"]:
                test_evaluator = pickle.load(open('test.evaluator', 'rb'))
            else:
                # TEST EVALUATOR
                test_evaluator = Evaluator(
                    pickle_source_path=params["test_dataset"]["pickle_source_path"],
                    set_subfolder=params["test_dataset"]["subset"],
                    hvo_pickle_filename=params["test_dataset"]["hvo_pickle_filename"],
                    list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
                    max_hvo_shape=(32, 27),
                    n_samples_to_use=11,
                    n_samples_to_synthesize_visualize_per_subset=10,
                    disable_tqdm=False,
                    analyze_heatmap=True,
                    analyze_global_features=True,
                    _identifier="Test_Set"
                )
            test_evaluator_subset = test_evaluator.get_ground_truth_hvo_sequences()
            metadata_test = pd.read_csv(os.path.join(params["test_dataset"]["pickle_source_path"],
                                                     params["test_dataset"]["subset"],
                                                     params["test_dataset"]["metadata_csv_filename"]))
            print("\nGenerating inputs for test evaluator...")
            test_eval_inputs, test_eval_gt, _ = process_dataset(test_evaluator_subset, metadata=metadata_test,
                                                                                          max_len=params["test_dataset"]["max_len"],
                                                                                          tappify_params=params["tappify_params"])
            print("Inputs for test evaluator generated.")


    # GENERATE FREQUENCY LOG ARRAYS
    epoch_save_partial, epoch_save_all = eval_log_freq(total_epochs=eps, initial_epochs_lim=10, initial_step_partial=1,
                                                       initial_step_all=1, secondary_step_partial=5,
                                                       secondary_step_all=5)

    # ONLY EVAL ON LAST EPOCH
    # epoch_save_partial, epoch_save_all = [eps-1], [eps-1]
    print("\nPartial evaluation saved on epoch(s) ", str(epoch_save_partial))
    print("Full evaluation saved on epoch(s) ", str(epoch_save_all))
    print("\nTraining model...")
    try:
        for i in np.arange(eps):
            ep += 1
            recalculate_gt = True if ep == 1 else False
            save_model = (i in epoch_save_partial or i in epoch_save_all)

            print(f"\nEpoch {ep}\n-------------------------------")
            train_loop(dataloader=dataloader, groove_transformer=model, opt=optimizer, epoch=ep, loss_fn=calculate_loss,
                       bce_fn=BCE_fn, mse_fn=MSE_fn, save=save_model, device=params["model"]["device"],
                       encoder_only=params["model"]["encoder_only"], hit_loss_penalty=params["model"]["loss_hit_penalty_multiplier"],
                       test_inputs=test_eval_inputs, test_gt=test_eval_gt)
            print("-------------------------------\n")

            if i in epoch_save_partial or i in epoch_save_all:
                if params["train_eval"]:
                    # EVAL TRAIN
                    # --------------------------------------------------------------------------------------------------
                    train_evaluator._identifier = 'Train_Set'
                    train_eval_pred = torch.cat(model.predict(train_eval_inputs, use_thres=True, thres=0.5), dim=2)
                    train_eval_pred_hvo_array = train_eval_pred.cpu().detach().numpy()
                    train_evaluator.add_predictions(train_eval_pred_hvo_array)

                    # Evaluate
                    train_acc_h = train_evaluator.get_hits_accuracies(drum_mapping=ROLAND_REDUCED_MAPPING)
                    train_mse_v = train_evaluator.get_velocity_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                    train_mse_o = train_evaluator.get_micro_timing_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                    # train_rhythmic_distances = train_evaluator.get_rhythmic_distances()

                    # Log
                    wandb.log(train_acc_h, commit=False)
                    wandb.log(train_mse_v, commit=False)
                    wandb.log(train_mse_o)
                    # wandb.log(train_rhythmic_distances, commit=False)

                    if i in epoch_save_all:
                        train_evaluator._identifier = 'Train_Set_Epoch_{}'.format(ep)
                        # Heatmaps train
                        train_heatmaps_global_features = train_evaluator.get_wandb_logging_media(
                            sf_paths=["../../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"],
                            recalculate_ground_truth=recalculate_gt)
                        if len(train_heatmaps_global_features.keys()) > 0:
                            wandb.log(train_heatmaps_global_features, commit=False)

                    train_evaluator.dump(path="misc/train_set_evaluator_run_{}_Epoch_{}.Eval".format(wandb_run.name, ep))
                    #---------------------------------------------------------------------------------------------------
                    wandb.log({"epoch": ep})

                if params["test_eval"]:
                    # EVAL TEST
                    #---------------------------------------------------------------------------------------------------
                    test_evaluator._identifier = 'Test_Set'
                    test_eval_pred = torch.cat(model.predict(test_eval_inputs, use_thres=True, thres=0.5), dim=2)
                    test_eval_pred_hvo_array = test_eval_pred.cpu().detach().numpy()
                    test_evaluator.add_predictions(test_eval_pred_hvo_array)

                    # Evaluate
                    test_acc_h = test_evaluator.get_hits_accuracies(drum_mapping=ROLAND_REDUCED_MAPPING)
                    test_mse_v = test_evaluator.get_velocity_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                    test_mse_o = test_evaluator.get_micro_timing_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                    # rhythmic_distances = test_evaluator.get_rhythmic_distances()

                    # Log
                    wandb.log(test_acc_h, commit=False)
                    wandb.log(test_mse_v, commit=False)
                    wandb.log(test_mse_o)
                    # wandb.log(rhythmic_distances, commit=False)

                    if i in epoch_save_all:
                        test_evaluator._identifier = 'Test_Set_Epoch_{}'.format(ep)
                        # Heatmaps test
                        test_heatmaps_global_features = test_evaluator.get_wandb_logging_media(
                            sf_paths=["../../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"],
                            recalculate_ground_truth=recalculate_gt)
                        if len(test_heatmaps_global_features.keys()) > 0:
                            wandb.log(test_heatmaps_global_features, commit=False)

                    test_evaluator.dump(path="misc/test_set_evaluator_run_{}_Epoch_{}.Eval".format(wandb_run.name, ep))
                    #---------------------------------------------------------------------------------------------------
                    wandb.log({"epoch": ep})
    finally:
        print("\nDone!")
        wandb.finish()
