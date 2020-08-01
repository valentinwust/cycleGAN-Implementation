import argparse
import cv2
from PIL import Image

class TrainOptions():
    def __init__(self):
        parser = argparse.ArgumentParser()
        
        # Epochs
        parser.add_argument("--epoch", type=int, default=0, help="start training at epoch")
        parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs with normal learning rate")
        parser.add_argument("--n_epochs_decay", type=int, default=100, help="number of epochs over which learning rate decays to zero")
        # Optimizer
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        # Channels
        parser.add_argument('--n_input', type=int, default=3, help='# of input image channels')
        parser.add_argument('--n_output', type=int, default=3, help='# of output image channels')
        
        # Parse arguments
        self.opt = parser.parse_args()


def save_image(path,image):
    cv2.imwrite(path, image)

def tensor_to_image(image):
    image_numpy = image.cpu().detach().permute(0,2,3,1).numpy()
    image_numpy = (image_numpy[0]+1)/2 * 255.0
    
    return image_numpy