import sys
import wandb
import numpy as np
import pandas as pd
import torch
from dataset import GrooveMidiDataset, process_dataset

sys.path.insert(1, '../../BaseGrooveTransformers/')
sys.path.insert(1, '../BaseGrooveTransformers/')
from models.train import *

sys.path.insert(1, '../../GrooveEvaluator/')
sys.path.insert(1, '../GrooveEvaluator/')
from GrooveEvaluator.evaluator import Evaluator

sys.path.insert(1, '../../preprocessed_dataset/')
sys.path.insert(1, '../preprocessed_dataset/')
from Subset_Creators.subsetters import GrooveMidiSubsetter

#import os
#os.environ['WANDB_MODE']='offline'

if __name__ == "__main__":

    hyperparameter_defaults = dict(
        optimizer_algorithm='sgd',
        d_model=128,
        n_heads=8,
        dropout=0.1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        learning_rate=1e-3,
        batch_size=64,
        dim_feedforward=1280,
        epochs=100,
        lr_scheduler_step_size=30,
        lr_scheduler_gamma=0.1
    )

    wandb_run = wandb.init(config=hyperparameter_defaults)
    
    save_info = {
        'checkpoint_path': '../results/',
        'checkpoint_save_str': '../results/transformer_groove_tap2drum-epoch-{}',
        'df_path': '../results/losses_df/'
    }

    filters = {
        "beat_type": ["beat"],
        "time_signature": ["4-4"]
    }

    subset_info = {
        "pickle_source_path": '../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.4'
                              '/Processed_On_09_06_2021_at_12_41_hrs',
        "subset": 'GrooveMIDI_processed_train',
        "metadata_csv_filename": 'metadata.csv',
        "hvo_pickle_filename": 'hvo_sequence_data.obj',
        "filters": filters
    }

    tap_params = {
        "tapped_sequence_voice": 'HH_CLOSED',
        "tapped_sequence_collapsed": False,
        "tapped_sequence_velocity_mode": 1,
        "tapped_sequence_offset_mode": 3
    }

    seq_max_len = 32

    # TRANSFORMER MODEL PARAMETERS
    model_parameters = {
        'optimizer': wandb.config.optimizer_algorithm,
        'd_model': wandb.config.d_model,
        'n_heads': wandb.config.n_heads,
        'dim_feedforward': wandb.config.dim_feedforward,
        'dropout': wandb.config.dropout,
        'num_encoder_layers': wandb.config.num_encoder_layers,
        'num_decoder_layers': wandb.config.num_decoder_layers,
        'max_len': 32,
        'embedding_size_src': 27,
        'embedding_size_tgt': 27,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # TRAINING PARAMETERS
    training_parameters = {
        'learning_rate': wandb.config.learning_rate,
        'batch_size': wandb.config.batch_size,
        'lr_scheduler_step_size': wandb.config.lr_scheduler_step_size,
        'lr_scheduler_gamma': wandb.config.lr_scheduler_gamma
    }

    # PYTORCH LOSS FUNCTIONS
    BCE_fn = torch.nn.BCEWithLogitsLoss()
    MSE_fn = torch.nn.MSELoss()

    # DATASET LOADING FOR TRAINING
    _, subset_list = GrooveMidiSubsetter(pickle_source_path=subset_info["pickle_source_path"],
                                         subset=subset_info["subset"],
                                         hvo_pickle_filename=subset_info["hvo_pickle_filename"],
                                         list_of_filter_dicts_for_subsets=[filters]).create_subsets()

    gmd = GrooveMidiDataset(subset=subset_list[0], subset_info=subset_info, max_len=seq_max_len,
                            tappify_params=tap_params)
    dataloader = DataLoader(gmd, batch_size=training_parameters['batch_size'], shuffle=True)

    # EVALUATOR
    evaluator = Evaluator(
        pickle_source_path=subset_info["pickle_source_path"],
        set_subfolder=subset_info["subset"],
        hvo_pickle_filename=subset_info["hvo_pickle_filename"],
        list_of_filter_dicts_for_subsets=[filters],
        max_hvo_shape=(32, 27),
        n_samples_to_use=1024,
        n_samples_to_synthesize_visualize_per_subset=10,
        disable_tqdm=False,
        analyze_heatmap=True,
        analyze_global_features=True
    )
    evaluator_subset = evaluator.get_ground_truth_hvo_sequences()
    metadata = pd.read_csv(os.path.join(subset_info["pickle_source_path"], subset_info["subset"],
                                            subset_info["metadata_csv_filename"]))
    eval_inputs, _, _ = process_dataset(evaluator_subset, metadata=metadata, max_len=seq_max_len,
                                                               tappify_params=tap_params)

    model, optimizer, scheduler, ep = initialize_model(model_parameters, training_parameters, save_info, load_from_checkpoint=False)
    wandb.watch(model)

    epoch_save_div = 1
    eps = wandb.config.epochs

    # GENERATE FREQUENCY LOG ARRAYS
    first_epochs_step = 1
    first_epochs_lim = 10 if eps >= 10 else eps
    epoch_save_partial = np.arange(first_epochs_lim, step=first_epochs_step)
    epoch_save_all = np.arange(first_epochs_lim, step=first_epochs_step)
    if first_epochs_lim != eps:
        remaining_epochs_step_partial, remaining_epochs_step_all = 5, 10
        epoch_save_partial = np.append(epoch_save_partial,
                                       np.arange(start=first_epochs_lim, step=remaining_epochs_step_partial, stop=eps))
        epoch_save_all = np.append(epoch_save_all,
                                   np.arange(start=first_epochs_lim, step=remaining_epochs_step_all, stop=eps))

    try:
        for i in np.arange(eps):
            ep += 1
            save_model = (i in epoch_save_partial or i in epoch_save_all)

            print(f"Epoch {ep}\n-------------------------------")
            train_loop(dataloader=dataloader, groove_transformer=model, opt=optimizer, scheduler=scheduler, epoch=ep,
                       loss_fn=calculate_loss, bce_fn=BCE_fn, mse_fn=MSE_fn, save_model=save_model, cp_info=save_info,
                       device=model_parameters['device'], wandb_run=wandb_run.name)
            print("-------------------------------\n")

            eval_pred = model.predict(eval_inputs, use_thres=True, thres=0.5)
            eval_pred_hvo_array = np.concatenate(eval_pred, axis=2)
            evaluator.add_predictions(eval_pred_hvo_array)

            if i in epoch_save_partial or i in epoch_save_all:

                # Evaluate
                acc_h = evaluator.get_hits_accuracies(drum_mapping=ROLAND_REDUCED_MAPPING)
                mse_v = evaluator.get_velocity_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                mse_o = evaluator.get_micro_timing_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
                rhythmic_distances = evaluator.get_rhythmic_distances()

                # Log
                wandb.log(acc_h, commit=False)
                wandb.log(mse_v, commit=False)
                wandb.log(mse_o, commit=False)
                wandb.log(rhythmic_distances, commit=False)

            if i in epoch_save_all:

                # Heatmaps
                heatmaps_global_features = evaluator.get_wandb_logging_media(
                    sf_paths=['../soundfonts/filtered_soundfonts/Standard_Drum_Kit.sf2'])
                if len(heatmaps_global_features.keys()) > 0:
                    wandb.log(heatmaps_global_features, commit=False)

            evaluator.dump(path="misc/evaluator_run_{}_Epoch_{}.Eval".format(wandb_run.name, ep))

    finally:
        wandb.finish()
