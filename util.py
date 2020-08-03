import argparse
import cv2
from PIL import Image
import os
import torchvision.transforms as transforms
import random
import torch.utils.data

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
    """ Function for saving images, needs to be adjusted for four-channel data!
    """
    cv2.imwrite(path, image)

def tensor_to_image(image):
    """ Returns the first image in image batch as RGB array
    """
    image_numpy = image.cpu().detach().permute(0,2,3,1).numpy()
    image_numpy = (image_numpy[0]+1)/2 * 255.0
    
    return image_numpy


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]

def get_image_paths(directory):
    files = sorted(os.listdir(directory))
    images = []
    for file in files:
        if any(file.endswith(extension) for extension in IMG_EXTENSIONS):
            images.append(directory + "/" + file)
    return images

class UnalignedDataset():
    def __init__(self, opt):
        
        self.opt = opt
        self.root = opt.dataroot
        
        self.dir_A = self.root+"/trainA"
        self.dir_B = self.root+"/trainB"
        
        self.A_paths = get_image_paths(self.dir_A)
        self.B_paths = get_image_paths(self.dir_B)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        ###### Add transformations!
        self.transform_A = transforms.ToTensor()
        self.transform_B = transforms.ToTensor()

    def __getitem__(self, index):
        """ 
        """
        
        A_path = self.A_paths[index] # Image A with index index
        B_path = self.B_paths[random.randint(0, self.B_size - 1)] # Random image from B to avoid fixed pairs
        
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)
        
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        
    def __len__(self):
        """ Return dataset size, take size of trainA
        """
        return self.A_size
        
class CustomDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = UnalignedDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True)#,
            #num_workers=int(opt.num_threads))
            ###### Multiprocessing does not work!?

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
    
    
    
    
    
    
    
    