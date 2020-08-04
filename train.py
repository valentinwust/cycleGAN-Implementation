import time
import torch
from tqdm import tqdm

import util
from model import CycleGANModel
from util import CustomDataLoader

""" 
    Missing:
        - different initializations?
        - argument for loading old model
        - 4-channel support for saving images, model etc.
    
"""


if __name__ == '__main__':
    opt = util.get_opt() # Does not work :(
    
    model = CycleGANModel(opt)
    if opt.load_model:
        model.load_model("latest_net_") # Load old model

    dataloader = CustomDataLoader(opt) # Dataloader, training loop with enumerate(dataloader) etc.
    print("Dataloader initialized")

    # ----------
    #  Training
    # ----------

    #initialize status bars and logging bars
    epoch_size = len(dataloader)
    epoch_bar = tqdm(
        range(opt.epoch, opt.n_epochs+opt.n_epochs_decay),
        position=1,
        )
    batch_bar = tqdm(
        enumerate(dataloader),
        position=2,
        )
    log_bar = tqdm(position=4, bar_format='{desc}')
    print("Logging bars initialized")

    def print_logs(**kwargs):
         model.print_logs(print_fn=log_bar.set_description_str)
         pass

    def call_breakpoint(**kwargs):
         breakpoint()
         pass

    #start hotkey instance
    hotkeys = util.Hotkey_handler()
    hotkeys.add_hotkey('s', model.save_model)
    hotkeys.add_hotkey('v', model.save_visuals)
    hotkeys.add_hotkey('u', print_logs)
    hotkeys.add_hotkey('b', call_breakpoint)
    print("Hotkeys initialized")

    prev_time = time.time()
    print("start loop")
    for epoch in epoch_bar:
        model.set_train()
        for i, batch in batch_bar:

            # -------------
            #  Train Model
            # -------------
                
            # Set model input
            model.set_inputs(batch)

            model.optimize()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i

            # If at sample interval save image
            batches_done = epoch * epoch_size + i
            if batches_done % opt.sample_interval == 0:
                model.save_visuals(idx=batches_done)
                model.print_logs(print_fn=log_bar.set_description_str)

            #key activated hotkeys and call their respective functions
            functions2call = hotkeys.get_function_list()
            for function in functions2call:
                function(epoch=epoch, idx=i)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            model.save_model(epoch=epoch)
