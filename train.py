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

def train(model):
    opt = util.get_opt()

    #add wandb to log images and losses
    
    dataloader = CustomDataLoader(opt) # Dataloader, training loop with enumerate(dataloader) etc.
    print("Dataloader initialized")

    def print_logs(**kwargs):
         model.print_logs(print_fn=log_bar.set_description_str)

    def call_breakpoint(**kwargs):
         breakpoint()

    #start hotkey instance
    hotkeys = util.Hotkey_handler()
    hotkeys.add_hotkey('s', model.save_model)
    hotkeys.add_hotkey('v', model.save_visuals)
    hotkeys.add_hotkey('u', print_logs)
    hotkeys.add_hotkey('b', call_breakpoint)
    hotkeys.add_hotkey('e', evaluate)
    print("Hotkeys initialized")

    #initialize status bars and logging bars
    epoch_bar = tqdm(
        range(opt.epoch, opt.n_epochs+opt.n_epochs_decay),
        position=1,
        )
    log_bar = tqdm(position=4, bar_format='{desc}')

    print("start loop")
    for epoch in epoch_bar:
        batch_bar = tqdm(
            enumerate(dataloader),
            position=2,
            total=len(dataloader),
            )
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

            # If at sample interval save image
            if model.step % opt.log_batch_freq == 0:
                model.print_logs(print_fn=log_bar.set_description_str)
            if model.step % opt.visual_batch_freq == 0:
                model.save_visuals()
            #if epoch % opt.eval_epoch_freq == 0:
            #    model.evaluate()

            #key activated hotkeys and call their respective functions
            functions2call = hotkeys.get_function_list()
            for function in functions2call:
                function()
        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            model.save_model()

def evaluate(model):
    opt = util.get_opt()

    #add wandb to log images and losses
    
    dataloader = CustomDataLoader(opt) # Dataloader, training loop with enumerate(dataloader) etc.
    print("Dataloader initialized")

    def print_logs(**kwargs):
         model.print_logs(print_fn=log_bar.set_description_str)

    def call_breakpoint(**kwargs):
         breakpoint()

    #start hotkey instance
    hotkeys = util.Hotkey_handler()
    hotkeys.add_hotkey('s', model.save_model)
    hotkeys.add_hotkey('v', model.save_visuals)
    hotkeys.add_hotkey('u', print_logs)
    hotkeys.add_hotkey('b', call_breakpoint)
    print("Hotkeys initialized")

    #initialize status bars and logging bars
    epoch_bar = tqdm(
        range(opt.epoch, opt.n_epochs+opt.n_epochs_decay),
        position=1,
        )
    log_bar = tqdm(position=4, bar_format='{desc}')

    print("start loop")
    for epoch in epoch_bar:
        batch_bar = tqdm(
            enumerate(dataloader),
            position=2,
            total=len(dataloader),
            )
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

            # If at sample interval save image
            if model.step % opt.log_batch_freq == 0:
                model.print_logs(print_fn=log_bar.set_description_str)
            if model.step % opt.visual_batch_freq == 0:
                model.save_visuals()
            #if epoch % opt.eval_epoch_freq == 0:
            #    model.evaluate()

            #key activated hotkeys and call their respective functions
            functions2call = hotkeys.get_function_list()
            for function in functions2call:
                function()
        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            model.save_model()



if __name__ == '__main__':

    model = CycleGANModel(opt)
    if opt.load_model:
        model.load_model("latest_net_") # Load old model

    # ----------
    #  Training
    # ----------

    train(model)

