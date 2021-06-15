import sys
import unittest
import torch
import wandb
from torch.utils.data import DataLoader
from model.dataset import GrooveMidiDatasetTap2Drum

sys.path.insert(1, "../../preprocessed_dataset/")
sys.path.insert(1, "../preprocessed_dataset/")
from Subset_Creators.subsetters import GrooveMidiSubsetter

sys.path.insert(1, "../../BaseGrooveTransformers/")
sys.path.insert(1, "../BaseGrooveTransformers/")
from models.train import initialize_model, train_loop, calculate_loss

import os
os.environ["WANDB_MODE"]="offline"

params = {
    "model": {
        "optimizer": "sgd",
        "d_model": 128,
        "n_heads": 8,
        "dim_feedforward": 512,
        "dropout": 0,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "max_len": 32,
        "embedding_size_src": 27,
        "embedding_size_tgt": 27,
        "encoder_only": True,  # Set to false for encoder-decoder
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "dataset": {
        "pickle_source_path": "../../preprocessed_dataset/datasets_extracted_locally/GrooveMidi/hvo_0.4.4"
                              "/Processed_On_09_06_2021_at_12_41_hrs",
        "subset": "GrooveMIDI_processed_train",
        "metadata_csv_filename": "metadata.csv",
        "hvo_pickle_filename": "hvo_sequence_data.obj",
        "filters": {
            "beat_type": ["beat"],
            "time_signature": ["4-4"],
            # "master_id": ["drummer1/session1/201"]  # 29 examples
            "master_id": ["drummer9/session1/8"]  # 4 examples
        },
        "max_len": 32
    },
    "tappify_params": {
        "tapped_sequence_voice": "HH_CLOSED",
        "tapped_sequence_collapsed": False,
        "tapped_sequence_velocity_mode": 1,
        "tapped_sequence_offset_mode": 3
    },
    "training": {
        "learning_rate": 0.1,
        "batch_size": 4,
        "lr_scheduler_step_size": 30,
        "lr_scheduler_gamma": 0.1
    },
    "load_model": None
    }


class Test_Transformer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, subset_list = GrooveMidiSubsetter(pickle_source_path=params["dataset"]["pickle_source_path"],
                                             subset=params["dataset"]["subset"],
                                             hvo_pickle_filename=params["dataset"]["hvo_pickle_filename"],
                                             list_of_filter_dicts_for_subsets=[
                                                 params["dataset"]["filters"]]).create_subsets()

        gmd = GrooveMidiDatasetTap2Drum(subset=subset_list[0], subset_info=params["dataset"],
                                        tappify_params=params["tappify_params"], max_len=params["dataset"]["max_len"])

        cls.dataloader = DataLoader(gmd, batch_size=params["training"]["batch_size"], shuffle=True)
        cls.model, cls.optimizer, cls.ep = initialize_model(params)
        cls.BCE_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        cls.MSE_fn = torch.nn.MSELoss(reduction='none')

        wandb.init()


    def test_parameters_one_epoch(self):
        """
        Test if after one epoch the parameters of the model change
        """
        model = Test_Transformer.model
        opt = Test_Transformer.optimizer
        scheduler = Test_Transformer.scheduler
        ep = Test_Transformer.ep + 1

        model_params = [np for np in model.named_parameters() if np[1].requires_grad]

        # make a copy of initial params
        initial_params = [(name, p.clone()) for (name, p) in model_params]

        train_loop(dataloader=Test_Transformer.dataloader, groove_transformer=model, opt=opt, scheduler=scheduler,
                   epoch=ep, loss_fn=calculate_loss, bce_fn=Test_Transformer.BCE_fn, mse_fn=Test_Transformer.MSE_fn,
                   save=False, device=params["model"]["device"], encoder_only=params["model"]["encoder_only"])

        for (_, p0), (name, p1) in zip(initial_params, model_params):
            if torch.equal(p0, p1):
                print("Param {} doesn't change in the first training step".format(name))
            self.assertFalse(torch.equal(p0, p1))

        return 0

    def test_parameters_const_loss(self):
        """
        Test if after running 500 epochs, loss is not changing (dropout=0) & if the parameters are the same from the
        first epoch where the loss is the same to the previous epoch
        """
        named_params = [np for np in Test_Transformer.model.named_parameters() if np[1].requires_grad]
        # print(sum(p.numel() for p in Test_Transformer.model.parameters() if p.requires_grad))  # number of parameters
        # copy initial parameters
        prev_params, last_params = [(name, p.clone()) for (name, p) in named_params], []

        model = Test_Transformer.model
        opt = Test_Transformer.optimizer
        # scheduler = Test_Transformer.scheduler
        ep = Test_Transformer.ep

        prev_loss, loss = -1, -2
        max_ep =2000

        while (prev_loss != loss and ep < max_ep):
            ep += 1
            print("Epoch {} -------------------------------\n".format(ep))
            prev_loss = loss
            prev_params = [(name, p.clone()) for (name, p) in named_params]
            loss = train_loop(dataloader=Test_Transformer.dataloader, groove_transformer=model, opt=opt,
                              epoch=ep, loss_fn=calculate_loss, bce_fn=Test_Transformer.BCE_fn,
                              mse_fn=Test_Transformer.MSE_fn, save=False, device=params["model"]["device"],
                              encoder_only=params["model"]["encoder_only"])
            last_params = [(name, p.clone()) for (name, p) in named_params]

        if ep == 2000:
            print("Loss is never the same (keeps changing from epoch to epoch)")
        else:
            change = True
            for (_, p0), (name, p1) in zip(prev_params, last_params):
                if torch.allclose(p0, p1, atol=0.000001):
                    print("Param {} doesn't change from epoch {} to {}".format(name, ep-1, ep))
                    change = False
            self.assertTrue(change)
