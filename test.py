import time
import torch
from tqdm import tqdm

import util
from model import GeneratorTestModel
from util import TestDataset

def test():
    opt = util.get_opt()
    
    model = GeneratorTestModel(opt)
    model.load_model(opt.load_model_name) # Load old model

    dataset = TestDataset(opt)
    print("Dataloader initialized")

    for image in tqdm(dataset):
        model.save_generated(image)
    



if __name__ == '__main__':

    # ----------
    #  Training
    # ----------

    test()

