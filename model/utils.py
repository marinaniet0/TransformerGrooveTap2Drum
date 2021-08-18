import numpy as np
import sys
from tqdm import tqdm
from bokeh.models.widgets import Tabs, Panel
from bokeh.embed import file_html
from bokeh.resources import CDN
import wandb
# sys.path.insert(1, "../../hvo_sequence/")
# sys.path.insert(1, "../hvo_sequence/")
from hvo_sequence.hvo_seq import *

def eval_log_freq(total_epochs, initial_epochs_lim, initial_step_partial, initial_step_all, secondary_step_partial,
                  secondary_step_all):

    if initial_epochs_lim >= total_epochs:
        epoch_save_partial = np.arange(total_epochs, step=initial_step_partial)
        epoch_save_all = np.arange(total_epochs, step=initial_step_all)
        return epoch_save_partial, epoch_save_all

    epoch_save_partial = np.arange(initial_epochs_lim, step=initial_step_partial)
    epoch_save_all = np.arange(initial_epochs_lim, step=initial_step_all)
    epoch_save_partial = np.append(epoch_save_partial, np.arange(start=initial_epochs_lim, step=secondary_step_partial,
                                                                 stop=total_epochs))
    epoch_save_all = np.append(epoch_save_all, np.arange(start=initial_epochs_lim, step=secondary_step_all,
                                                         stop=total_epochs))
    if total_epochs-1 not in epoch_save_partial:
        epoch_save_partial = np.append(epoch_save_partial, total_epochs-1)
    if total_epochs-1 not in epoch_save_all:
        epoch_save_all = np.append(epoch_save_all, total_epochs-1)
    return epoch_save_partial, epoch_save_all

def get_samples_hvo_eval(evaluator):
    tags, subsets = evaluator._gt_tags, evaluator._gt_subsets
    sampled_hvos = {x: [] for x in tags}
    for subset_ix, tag in enumerate(tags):
        sampled_hvos[tag] = [subsets[subset_ix][ix] for ix in evaluator.audio_sample_locations[tag]]

    return sampled_hvos

def update_dict_with_tapped(sf_path, evaluator, evaluator_id, wandb_dict):
    sampled_hvos = get_samples_hvo_eval(evaluator)

    # AUDIOS
    tapped_audios = []
    tapped_captions = []
    for key in tqdm(sampled_hvos.keys(),
                    desc='Synthesizing samples - {}_Set_Tapped'.format(evaluator_id)):
        for sample_hvo in sampled_hvos[key]:
            tapped_hvo = zero_like(sample_hvo)
            tapped_hvo.hvo = sample_hvo.flatten_voices()
            tapped_audios.append(tapped_hvo.synthesize(sf_path=sf_path))
            tapped_captions.append("{}_{}_{}.wav".format(
                "{}_Set_Tapped".format(evaluator_id), sample_hvo.metadata.style_primary,
                sample_hvo.metadata.master_id.replace("/", "_")))
    # sort so that they are alphabetically ordered in wandb
    sort_index = np.argsort(tapped_captions)
    tapped_captions = np.array(tapped_captions)[sort_index].tolist()
    tapped_audios = np.array(tapped_audios)[sort_index].tolist()
    tapped_captions_audios_tuples = list(zip(tapped_captions, tapped_audios))
    tapped_captions_audios = [(c_a[0], c_a[1]) for c_a in tapped_captions_audios_tuples]
    wandb_dict["audios"].update({"{}_Set_Tapped".format(evaluator_id): [
        wandb.Audio(c_a[1], caption=c_a[0], sample_rate=44100) for c_a in tapped_captions_audios]})

    # PIANO ROLLS

    tab_titles = []
    piano_roll_tabs = []
    for subset_ix, tag in tqdm(enumerate(sampled_hvos.keys()),
                               desc='Creating Piano rolls for {}_Set_Tapped'.format(evaluator_id)):
        piano_rolls = []
        for sample_hvo in sampled_hvos[tag]:
            tapped_hvo = zero_like(sample_hvo)
            tapped_hvo.hvo = sample_hvo.flatten_voices()
            title = "{}_{}_{}".format(
                "{}_Set_Tapped".format(evaluator_id), sample_hvo.metadata.style_primary,
                sample_hvo.metadata.master_id.replace("/", "_"))
            piano_rolls.append(tapped_hvo.to_html_plot(filename=title))
        piano_roll_tabs.append(separate_figues_by_tabs(piano_rolls, [str(x) for x in range(len(piano_rolls))]))
        tab_titles.append(tag)

    # sort so that they are alphabetically ordered in wandb
    sort_index = np.argsort(tab_titles)
    tab_titles = np.array(tab_titles)[sort_index].tolist()
    piano_roll_tabs = np.array(piano_roll_tabs)[sort_index].tolist()

    rolls = separate_figues_by_tabs(piano_roll_tabs, [tag for tag in tab_titles])
    wandb_dict["piano_roll_html"].update({"{}_Set_Tapped".format(evaluator_id):
                        wandb.Html(file_html(rolls, CDN, "piano_rolls_" + "{}_Set_Tapped".format(evaluator_id)))})

    return wandb_dict


def separate_figues_by_tabs(bokeh_fig_list, tab_titles=None, top_panel_identifier="::"):

    titles = [str(tab_ix) for tab_ix in range(bokeh_fig_list)] if tab_titles is None else tab_titles

    top_tab_bottom_tabs_dicts = {"other": []}

    for ix, tab_title in enumerate(tab_titles):
        _p = bokeh_fig_list[ix]
        if len(tab_title.split("::")) > 1:
            top_key = tab_title.split("::")[0]
            title_ = tab_title.split("::")[1]
            if top_key not in top_tab_bottom_tabs_dicts.keys():
                top_tab_bottom_tabs_dicts.update({top_key: [Panel(child=_p, title=title_)]})
            else:
                top_tab_bottom_tabs_dicts[top_key].append(Panel(child=_p, title=title_))
        else:
            title_ = tab_title
            top_tab_bottom_tabs_dicts["other"].append(Panel(child=_p, title=title_))

    top_panels = []
    for major_key in top_tab_bottom_tabs_dicts.keys():
        top_panels.append(Panel(child=Tabs(tabs=top_tab_bottom_tabs_dicts[major_key]), title=major_key))

    return Tabs(tabs=top_panels)
