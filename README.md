# ![Tap2Drum](./imgs/t2d.png) with Transformer Neural Networks
This project, developed in the framework of a Master Thesis, trains and evaluates a Transformer model using the Groove
Dataset for the task of Tap2Drum.  The project and many of the ideas that we have tried out are based on the work done
by the Magenta team in their [Groove project](https://magenta.tensorflow.org/datasets/groove) and paper [Learning to
Groove with Inverse Sequence Transformations](https://arxiv.org/abs/1905.06118), including the very own definition of
the Tap2Drum task.

The repo has been developed along (and depends on) these projects (the links might not work for the moment, as
they are in private mode, but will eventually be public):  
 - [hvo_sequence](https://github.com/behzadhaki/hvo_sequence) - HVO data structure definition (which is the data type we
  will use to train the model) & functions to transform between hvo, [note_seq](https://github.com/magenta/note-seq)
  and MIDI formats
 - [GMD2HVO_PreProcessing](https://github.com/behzadhaki/GMD2HVO_PreProcessing) - Used to extract 2-bar sequences from
  the [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove), convert them to HVO format and save the
  preprocessed dataset
 - [preprocessed_dataset](https://github.com/behzadhaki/preprocessed_dataset) - Zipped preprocessed datasets in HVO format
 - [BaseGrooveTransformers](https://github.com/behzadhaki/BaseGrooveTransformers) - Transformer model used
 - [GrooveEvaluator](https://github.com/behzadhaki/GrooveEvaluator) - Evaluator that calculates a number of metrics
   given some ground truths and predictions and is able to export html and audio files.

To track the learning and metrics of the model we have used the [wandb](https://wandb.ai/) library and API. You can find
all our experiment runs in this [wandb project](https://wandb.ai/marinaniet0/transformer_groove_tap2drum).


In order to get this project up and running if you want to do your own training, you should first make sure you have all
dependencies installed (check the [requirements](./requirements.txt) file) and then clone the aforementioned
repositories in the same folder as this one, following this structure:
```
project_name
├───BaseGrooveTransformers
│   └───...
├───GrooveEvaluator
│   └───...
├───GrooveMIDI2HVO_PreProcessing
│   └───...
├───hvo_sequence
│   └───...
├───preprocessed_dataset
│   └───...
└───TransformerGrooveTap2Drum
    └───...
```
Then, extract the zipped datasets by running `extract_locally.py` located in the `preprocessed_dataset` repo.

Now you should be ready to run the training script located in this repository,
[`model/train_tap2drum.py`](./model/train_tap2drum.py).

One more thing! If you would like to track your model in wandb, make sure you have logged in and change
your project name/config to your own in [`model/train_tap2drum.py`](./model/train_tap2drum.py) line 47:
```python
wandb.init(config="configs/myconfig.yaml", project="myproject")
```
If you don't want to upload any data to wandb, you can uncomment lines 23 and 24:
```python
import os
os.environ["WANDB_MODE"]="offline"
```

And that should work!

Currently working on documentation, packaging the repos (to make this setup easier) and making a notebook to show how
to use the trained model.

Special thanks to Behzad Haki and Teresa Pelinski for all their work and help! :sparkles:
