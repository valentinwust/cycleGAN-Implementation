import argparse
import cv2
from PIL import Image
import threading

import os
import torchvision.transforms as transforms
import random
import torch.utils.data
import numpy as np

def get_opt():
    parser = argparse.ArgumentParser()
    
    # Epochs
    parser.add_argument("--epoch", type=int, default=0, help="start training at epoch")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs with normal learning rate")
    parser.add_argument("--n_epochs_decay", type=int, default=100, help="number of epochs over which learning rate decays to zero")
    parser.add_argument("--log_batch_freq", type=int, default=10, help="evaluation frequency")
    parser.add_argument("--visual_batch_freq", type=int, default=1000, help="evaluation frequency")
    parser.add_argument("--save_epoch_freq", type=int, default=10, help="evaluation frequency")
    parser.add_argument("--eval_epoch_freq", type=int, default=1, help="evaluation frequency")
    # Optimizer
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--batch_size', type=int, default=1, help='size of mini-batches')
    # Channels
    parser.add_argument('--n_input', type=int, default=3, help='# of input image channels')
    parser.add_argument('--n_output', type=int, default=3, help='# of output image channels')
    parser.add_argument('--forward_mask', action='store_true', help='whether to forward the mask and concatenate it to the output')

    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='ids for GPUs')
    parser.add_argument('--checkpoints_dir', type=str, default="checkpoints", help='path to checkpoint dir')
    parser.add_argument('--name', type=str, default='cycleGAN', help='name of model, e.g. star_witcher')
    parser.add_argument('--dataroot', type=str, default="datasets/star_witcher_data", help='path to data set')
    parser.add_argument('--num_threads', type=int, default=4, help='number of parallel threads for dataloader')

    parser.add_argument('--load_model', action='store_true', help='load trained model?')
    parser.add_argument('--load_model_name', type=str, default="latest_net_", help='Name of load model')

    parser.add_argument('--serial_batches', action='store_true', help='whether dataloader should not randomly select images from B')
    parser.add_argument('--no_flip', action='store_true', help='prevent data augmentation (flipping images)')
    parser.add_argument('--preprocess', type=str, default='crop', help='cropping of images at load time [crop | none]')
    parser.add_argument('--crop_size', type=int, default=256, help='size of crop of training images')
    
    # Parse arguments
    opt = parser.parse_args()

    return opt


def save_image(path,image):
    """ Function for saving images, needs to be adjusted for four-channel data!
    """
    cv2.imwrite(path, image)

def tensor_to_image(image,index=0):
    """ Returns the first image in image batch as RGB array
    """
    image_numpy = image.cpu().detach().permute(0,2,3,1).numpy()
    image_numpy = (image_numpy[index]+1)/2 * 255.0
    
    return image_numpy


# Multi-resolution grid

def draw_multires_figure(images, n_columns, imageres=(1920, 1080)):
    #check whter its enough iamges for columns or take the first n images
    imageres = np.shape(images)[1:3]
    canvas = Image.new('RGB', (imageres[1] * 2 - imageres[1]//(2**n_columns), imageres[0] * 2), 'white') #pich the right resolution
    n_images = 2 ** (n_columns + 1) - 2
    images = np.tile(images, (n_images//images.shape[0] + 1, 1, 1, 1))
    image_iter = iter(list(images))
    for col in range(n_columns):
        res_fraction = 2**(col)
        for row in range(2 ** (col+1)):
            image = next(image_iter)
            try:
                image = Image.fromarray(image.astype(np.uint8))
            except:
                breakpoint()
            image = image.resize((imageres[1] // res_fraction, imageres[0] // res_fraction), Image.ANTIALIAS)
            canvas.paste(image, (2 * imageres[1] - 2 * (imageres[1] // res_fraction), row * (imageres[0] // res_fraction)))
    return canvas

class Hotkey_handler(threading.Thread):
    def __init__(self, name='keyboard-input-thread'):
        super(Hotkey_handler, self).__init__(name=name)
        self.hotkeys = {} #stires data as pressed key: function to call
        self.functions2call = [] #list of functions to call because their key has been pressed
        self.start()

    def run(self):
        while True:
            self.on_press(input()) #waits to get input + Return

    def on_press(self, inp):
        try:
            if inp in self.hotkeys.keys():
                function = self.hotkeys[inp]
                self.functions2call.append(function)
                #pass
                #print(f'hotkey {key.char} pressed call function {function.__name__}', end="")
            else:
                pass
                #print(f'hotkey {key.char} not known', end="")

        except AttributeError:
            pass
            #print(f'special key {key} pressed')

    def add_hotkey(self, key, function):
        try:
            assert len(key) == 1
            self.hotkeys.update({key: function})
            print(f'hotkey {key} will call function {function.__name__}')
        except:
            print(f"new hotkey {key} not recognized!")
            #self.listener.stop()
            #breakpoint()

    def get_function_list(self):
        function_list = self.functions2call
        self.functions2call = []
        return function_list


def ensure_existance_paths(opt):
    """ Make sure dataset exists, and create folders for saving stuff if necessary
    """
    # Image folders
    assert os.path.isdir(opt.dataroot+"/trainA")
    assert os.path.isdir(opt.dataroot+"/trainB")
    # Folders for saving stuff
    if not os.path.isdir(opt.checkpoints_dir +"/"+opt.name+"/images"):
        os.makedirs(opt.checkpoints_dir +"/"+opt.name+"/images", exist_ok=True)
    if not os.path.isdir(opt.checkpoints_dir +"/"+opt.name+"/models"):
        os.makedirs(opt.checkpoints_dir +"/"+opt.name+"/models", exist_ok=True)

def ensure_existance_paths_test(opt):
    """ Make sure dataset exists, and create folders for saving stuff if necessary
    """
    # Image folders
    assert os.path.isdir(opt.dataroot)
    # Folders for saving stuff
    if not os.path.isdir(opt.checkpoints_dir +"/"+opt.name+"/generated"):
        os.makedirs(opt.checkpoints_dir +"/"+opt.name+"/generated", exist_ok=True)


####################
# Transformations  #
####################

def get_transform(opt, method=Image.BICUBIC, convert=True):
    transform_list = []

    if 'crop' in opt.preprocess:
        transform_list.append(transforms.RandomCrop(opt.crop_size))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_base(img, base=4, method=method)))

    if not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if convert:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize([0.5]*opt.n_input, [0.5]*opt.n_input)]
    
    return transforms.Compose(transform_list)

def get_transform_eval(opt, method=Image.BICUBIC, convert=True):
    transform_list = []

    transform_list.append(transforms.Lambda(lambda img: __make_power_base(img, base=4, method=method)))

    if convert:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize([0.5]*opt.n_input, [0.5]*opt.n_input)]
    
    return transforms.Compose(transform_list)



def __resize(image, shape, method=Image.BICUBIC):
    bands = image.split()
    bands = [C.resize(shape,method) for C in bands]
    if len(bands) == 3:
        return Image.merge('RGB', bands)
    elif len(bands) == 4:
        return Image.merge('RGBA', bands)
    else:
        raise Exception('Image has neither three nor four channels!')

def __make_power_base(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return __resize(img,(w, h), method)

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


####################
#     Dataset      #
####################

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]

def get_image_paths(directory):
    assert os.path.isdir(directory)
    
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
        self.channels = opt.n_input
        
        self.dir_A = self.root+"/trainA"
        self.dir_B = self.root+"/trainB"
        
        self.A_paths = get_image_paths(self.dir_A)
        self.B_paths = get_image_paths(self.dir_B)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform_A = get_transform(opt)
        self.transform_B = get_transform(opt)

    def __getitem__(self, index):
        """ 
        """
        
        A_path = self.A_paths[index % self.A_size] # Image A with index index
        B_path = self.B_paths[index % self.B_size if self.opt.serial_batches else random.randint(0, self.B_size - 1)] # Random image from B to avoid fixed pairs if not serial_batches
        
        A_img = np.array(Image.open(A_path))[...,:self.channels]
        B_img = np.array(Image.open(B_path))[...,:self.channels]
        
        A = self.transform_A(Image.fromarray(A_img))
        B = self.transform_B(Image.fromarray(B_img))

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
            shuffle=True,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

class EvalDataset():
    def __init__(self, opt):
        
        self.opt = opt
        self.opt.no_flip = True
        self.root = opt.dataroot
        
        self.dirA = self.root + "/testA"
        self.dirB = self.root + "/testB"
        
        self.pathsA = get_image_paths(self.dirA)
        self.pathsB = get_image_paths(self.dirB)
        if len(self.pathsB) == len(self.pathsA):
            self.size = len(self.pathsA)
        else:
            self.size = min(len(self.pathsA), len(self.pathsB))
            self.pathsA = self.pathsA[:self.size]
            self.pathsB = self.pathsB[:self.size]
        
        self.transform = get_transform_eval(self.opt)

    def __getitem__(self, index):
        """ 
        """
        
        path = self.pathsA[index]
        name, file_type = os.path.splitext(os.path.basename(path))
        imgA = Image.open(path)

        path = self.pathsB[index]
        name, file_type = os.path.splitext(os.path.basename(path))
        imgB = Image.open(path)
        
        A = self.transform(imgA).unsqueeze(0)
        B = self.transform(imgB).unsqueeze(0)

        return {'A': A, 'B': B}
        
    def __len__(self):
        """ Return dataset size, take size of trainA
        """
        return self.size

class TestDataset():
    def __init__(self, opt):
        
        self.opt = opt
        self.opt.no_flip = True
        self.root = opt.dataroot
        
        self.dir_ = self.root
        
        self.paths = get_image_paths(self.dir_)
        self.size = len(self.paths)
        
        self.transform = get_transform(self.opt)

    def __getitem__(self, index):
        """ 
        """
        
        path = self.paths[index]
        name, file_type = os.path.splitext(os.path.basename(path))
        
        img = Image.open(path)
        
        A = self.transform(img).unsqueeze(0)

        return {'A': A, 'A_name': name}
        
    def __len__(self):
        """ Return dataset size, take size of trainA
        """
        return self.size
