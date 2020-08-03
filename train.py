from util import TrainOptions
from model import CycleGANModel, CustomDataLoader

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
    
    #model = CycleGANModel(opt) # Create Model
    #model.load_model("latest_net_") # Load old model
    #data = {"A":torch.ones((1,3,100,100)),"B":torch.zeros((1,3,100,100))} # Example for data
    #model.set_inputs(data) # Set input
    #model.optimize() # Do backprop etc.
    #model.save_visuals() # Save current visuals, i.e. real_A, fake_B etc.
    #dataloader = CustomDataLoader(opt) # Dataloader, training loop with enumerate(dataloader) etc.