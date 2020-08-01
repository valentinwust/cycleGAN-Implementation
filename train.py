from util import TrainOptions
from model import *

if __name__ == '__main__':
    opt = TrainOptions().opt
    
    model = CycleGANModel(opt)