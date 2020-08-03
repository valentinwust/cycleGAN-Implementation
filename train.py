from util import TrainOptions
from model import *
import time
from tqdm import tqdm, t_range

if __name__ == '__main__':
    opt = TrainOptions().opt
    
    model = CycleGANModel(opt)

    dataloader = # load dateloader here

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
