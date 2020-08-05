import argparse
import cv2
from PIL import Image
import threading

import os
import torchvision.transforms as transforms
import random
import torch.utils.data

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
    parser.add_argument('--forward_mask', aciton='store_true', help='whether to forward the mask and concatenate it to the output')

    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='ids for GPUs')
    parser.add_argument('--checkpoints_dir', type=str, default="checkpoints", help='path to checkpoint dir')
    parser.add_argument('--name', type=str, default='cycleGAN', help='name of model, e.g. star_witcher')
    parser.add_argument('--dataroot', type=str, default="datasets/star_witcher_data", help='path to data set')
    parser.add_argument('--num_threads', type=int, default=4, help='number of parallel threads for dataloader')

    parser.add_argument('--load_model', type=bool, default=False, help='load trained model?')

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

def tensor_to_image(image):
    """ Returns the first image in image batch as RGB array
    """
    image_numpy = image.cpu().detach().permute(0,2,3,1).numpy()
    image_numpy = (image_numpy[0]+1)/2 * 255.0
    
    return image_numpy


##----------------------------------------------------------------------------
## Figures 2, 3, 10, 11, 12: Multi-resolution grid of uncurated result images.
#
#def draw_uncurated_result_figure(png, Gs, cx, cy, cw, ch, rows, lods, seed):
#    print(png)
#    latents = np.random.RandomState(seed).randn(sum(rows * 2**lod for lod in lods), Gs.input_shape[1])
#    images = Gs.run(latents, None, **synthesis_kwargs) # [seed, y, x, rgb]
#
#    canvas = PIL.Image.new('RGB', (sum(cw // 2**lod for lod in lods), ch * rows), 'white')
#    image_iter = iter(list(images))
#    for col, lod in enumerate(lods):
#        for row in range(rows * 2**lod):
#            image = PIL.Image.fromarray(next(image_iter), 'RGB')
#            image = image.crop((cx, cy, cx + cw, cy + ch))
#            image = image.resize((cw // 2**lod, ch // 2**lod), PIL.Image.ANTIALIAS)
#            canvas.paste(image, (sum(cw // 2**lod for lod in lods[:col]), row * ch // 2**lod))
#    canvas.save(png)

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
            shuffle=True,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
