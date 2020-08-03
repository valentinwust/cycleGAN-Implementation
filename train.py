import time
from tqdm import tqdm, t_range
from util import TrainOptions, CustomDataLoader
from model import CycleGANModel

""" 
    Missing:
        - argument parser
        - training loop
        - different initializations?
        - argument for loading old model
        - Create folders if not existing
        - Transformations in data loader, crop etc.
        - Multiprocessing for dataloader
        - 4-channel support for saving images etc.
    
"""


if __name__ == '__main__':
    #opt = TrainOptions().opt # Does not work :(
    
    ##### Arguments that I use so far
    import argparse
    opt = argparse.ArgumentParser()
    opt.lr = 0.0002
    opt.beta1 = 0.5
    opt.n_input = 3
    opt.n_output = 3
    opt.epoch = 0
    opt.n_epochs = 100
    opt.n_epochs_decay = 100
    opt.gpu_ids = [0]
    opt.checkpoints_dir = "checkpoints"
    opt.name = "style_star_witcher_first"
    opt.dataroot = "datasets/star_witcher_data"
    opt.batch_size = 1
    opt.num_threads = 4
    opt.load_model=False
    
    model = CycleGANModel(opt)
    if opt.load_model:
        model.load_model("latest_net_") # Load old model

    dataloader = CustomDataLoader(opt) # Dataloader, training loop with enumerate(dataloader) etc.

    data = {"A":torch.ones((1,3,100,100)),"B":torch.zeros((1,3,100,100))} # Example for data

    # ----------
    #  Training
    # ----------

    #initialize status bars and logging bars
    epoch_size = len(dataloader)
    epoch_bar = tqdm.trange(
        opt.epoch,
        opt.n_epochs,
        position=1,
        )
    batch_bar = tqdm(
        enumerate(dataloader),
        position=2,
        )
    log_bar = tqdm.tqdm(position=4, bar_format='{desc}')

    #start hotkey instance
    hotkeys = utils.Hotkey_handler()
    hotkeys.add_hotkey('s', model.save_model)
    hotkeys.add_hotkey('v', model.save_visuals)
    hotkeys.add_hotkey('u', lambda x: model.print_logs(print_fn=log_bar.set_description_str))

    prev_time = time.time()
    for epoch in epoch_bar:
        for i, batch in batch_bar:

            # -------------
            #  Train Model
            # -------------
                
            model.train()
            # Set model input
            model.set_inputs = model(batch)

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
