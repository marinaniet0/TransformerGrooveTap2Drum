import numpy as np


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
    return epoch_save_partial, epoch_save_all
