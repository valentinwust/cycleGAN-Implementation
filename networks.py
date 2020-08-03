import torch
import torch.nn as nn
from torch.optim import lr_scheduler


def initialize_weights(layer):
    """ Initialize weights of layer with normal distribution,
        only for the convolutional layers as the rest has no weights
    """
    classname = layer.__class__.__name__
    if "Conv" in classname:
         nn.init.normal_(layer.weight.data, 0.0, 0.02)

def get_optimizer_scheduler(optimizer,opt):
    """ Return learning rate scheduler for optimizer
    """
    def rule(epoch):
        lr_lambda = 1.0 - max(0, epoch + opt.epoch - opt.n_epochs) / float(opt.n_epochs_decay)
        return lr_lambda
    
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)
    return scheduler

def to_device(net, gpu_ids):
    """ Send net to GPU and enable parallelization
    """
    if gpu_ids and torch.cuda.is_available():
        net.to(gpu_ids[0])
        net = nn.DataParallel(net, gpu_ids)
    return net

####################
#      ResNet      #
####################

class ResNetBlock(nn.Module):
    """ Single ResNet block """
    def __init__(self, n_c):
        """ 
        Parameters:
            n_c (int)   - number of input/output channels
        """
        super(ResNetBlock, self).__init__()
        
        # Two convolutional layers with padding, InstanceNorm, only one ReLU
        layers = [nn.ReflectionPad2d(1),
                  nn.Conv2d(n_c, n_c, kernel_size = 3),
                  nn.InstanceNorm2d(n_c),
                  nn.ReLU(True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(n_c, n_c, kernel_size = 3),
                  nn.InstanceNorm2d(n_c)]
        
        # Assemble model
        self.conv_block = nn.Sequential(*layers)
        
    def forward(self, x):
        """ Forward pass with skip connection """
        return x + self.conv_block(x)


class ResNetGenerator(nn.Module):
    """ ResNet based generator.
        
        Downsamples the image, runs it through residual blocks,
        then upsamples backt to original size.
    """
    def __init__(self, n_input, n_output, n_res_blocks=9, n_filters=64, n_down_up_sampling=2):
        """ 
        Parameters:
            n_input  (int)           - number of input channels
            n_output (int)           - number of output channels
            n_res_blocks (int)       - number of residual blocks in between downsampling/upsampling layers
            n_filters (int)          - number of filters in the first and last conv layers
            n_down_up_sampling (int) - number of downsampling/upsampling layers
        """
        assert n_res_blocks > 0 # Just in case
        super(ResNetGenerator, self).__init__()
        
        # First convolutional layer
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(n_input, n_filters, kernel_size = 7),
                 nn.InstanceNorm2d(n_filters),
                 nn.ReLU(True)]
        
        # Downsampling layers
        for i in range(n_down_up_sampling):
            factor = 2**i
            model += [nn.Conv2d(n_filters * factor, n_filters * factor * 2, kernel_size = 3, stride=2, padding=1),
                      nn.InstanceNorm2d(n_filters * factor * 2),
                      nn.ReLU(True)]
        
        # ResNet blocks
        factor = 2**n_down_up_sampling
        for i in range(n_res_blocks):
            model += [ResNetBlock(n_filters * factor)]
        
        # Upsampling layers
        for i in range(n_down_up_sampling):
            factor = 2**(n_down_up_sampling-i)
            model += [nn.ConvTranspose2d(n_filters * factor, int(n_filters * factor / 2), kernel_size = 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(n_filters * factor / 2)),
                      nn.ReLU(True)]
        
        # Last convolutional layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(n_filters, n_output, kernel_size = 7),
                  nn.Tanh()]
        
        # Assemble model
        self.model = nn.Sequential(*model)
        
        # Initialize weights
        self.model.apply(initialize_weights)
    
    def forward(self, x):
        """ Forward pass """
        return self.model(x)


####################
#   Discriminator  #
####################

class NLayerDiscriminator(nn.Module):
    """ PatchGAN discriminator
        
        Receptive field (for n_layers=3):
            1. conv:   4x4
            2. conv:  10x10
            3. conv:  22x22
            4. conv:  46x46
            5. conv:  70x70
        
        Every output neuron classifies a 70x70 patch, the total
        classification is the mean over the last layer.
    """

    def __init__(self, n_input, n_filters=64, n_layers=3):
        """
        Parameters:
            n_input (int)   - number of channels in input images
            n_filters (int) - number of filters in the first conv layer
            n_layers (int)  - number of conv layers
        """
        assert n_layers > 0 # Just in case
        super(NLayerDiscriminator, self).__init__()
        
        # First conv layer without instance norm
        layers = [nn.Conv2d(n_input, n_filters, kernel_size = 4, stride = 2, padding = 1),
                  nn.LeakyReLU(0.2, True)]
        
        # Further conv layers with norm
        factor = 1
        factor_prev = 1
        for n in range(1,n_layers):
            factor_prev = factor
            factor = min(2**n, 8)
            layers += [nn.Conv2d(n_filters * factor_prev, n_filters * factor, kernel_size = 4, stride = 2, padding = 1),
                       nn.InstanceNorm2d(n_filters * factor),
                       nn.LeakyReLU(0.2, True)]
        
        # Another conv layer with reduced stride
        factor_prev = factor
        factor = min(2**n_layers, 8)
        layers += [nn.Conv2d(n_filters * factor_prev, n_filters * factor, kernel_size = 4, stride = 1, padding = 1),
                   nn.InstanceNorm2d(n_filters * factor),
                   nn.LeakyReLU(0.2, True)]
        
        # Last conv layer to reduce to one channel prediction map
        layers += [nn.Conv2d(n_filters * factor, 1, kernel_size = 4, stride = 1, padding = 1)]
        
        # Assemble model
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.model.apply(initialize_weights)
        
    def forward(self, x):
        """ Forward pass """
        return self.model(x)


####################
#      GAN loss    #
####################

class GANLoss(nn.Module):
    """ GAN loss
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """ 
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        """ Create label tensors
        """
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """ Calculate loss
        """
        loss = self.loss(prediction, self.get_target_tensor(prediction, target_is_real))
        return loss
