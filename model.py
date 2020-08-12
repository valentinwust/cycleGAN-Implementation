import torch
import torch.nn as nn
import itertools
import networks
import os
from util import save_image, tensor_to_image, ensure_existance_paths, ensure_existance_paths_test
import util
from tqdm import tqdm
import numpy as np

import wandb


####################
#     CycleGAN     #
####################

class CycleGANModel():
    def __init__(self, opt):
        """ 
        Parameters:
        """
        super(CycleGANModel, self).__init__()
        
        # Stuff
        ensure_existance_paths(opt) # Create necessary directories for saving stuff
        
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids and torch.cuda.is_available() else torch.device('cpu')
        self.lambda_idt = 0.5
        self.loss_lambda = 10.0
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'idt_B', 'real_B', 'fake_A', 'rec_B', 'idt_A']
        self.name = self.opt.name
        
        # Define models
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        self.netG_A = networks.to_device(networks.ResNetGenerator(opt.n_input, opt.n_output, opt.forward_mask),self.gpu_ids)
        self.netG_B = networks.to_device(networks.ResNetGenerator(opt.n_input, opt.n_output, opt.forward_mask),self.gpu_ids)
        self.netD_A = networks.to_device(networks.NLayerDiscriminator(opt.n_output),self.gpu_ids)
        self.netD_B = networks.to_device(networks.NLayerDiscriminator(opt.n_output),self.gpu_ids)
        
        # Define losses
        self.criterionGAN = networks.GANLoss().to(self.device)
        self.criterionCycle = nn.L1Loss().to(self.device)
        self.criterionIdt = nn.L1Loss().to(self.device)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        
        # Define optimizers
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_D)
        
        # Define schedulers
        self.schedulers = [networks.get_optimizer_scheduler(optimizer,opt) for optimizer in self.optimizers]

        self.set_train()

        self.step = 0

        wandb.init(
            entity="star-witchers",
            project="cycleGAN",
            config=self.opt,
            name=self.name,
            ) 

    def set_inputs(self,inputs):
        """ Set input images from dict inputs
        Parameters:
            
        """
        self.real_A = inputs["A"].to(self.device)
        self.real_B = inputs["B"].to(self.device)
    
    def forward(self):
        """ Perform forward pass
        """
        # Normal forward pass
        self.fake_B = self.netG_A(self.real_A) # G_A(A)
        self.fake_A = self.netG_B(self.real_B) # G_B(B)
        # Cycle
        self.rec_A = self.netG_B(self.fake_B) # G_B(G_A(A))
        self.rec_B = self.netG_A(self.fake_A) # G_A(G_B(B))
        # Identity
        self.idt_B = self.netG_B(self.real_A) # G_B(A)
        self.idt_A = self.netG_A(self.real_B) # G_A(B)
    
    def backward_G(self):
        """ Loss for the generators
        """
        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B),True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A),True)
        # Cycle loss
        self.loss_cycle_A = self.loss_lambda * self.criterionCycle(self.rec_A,self.real_A)
        self.loss_cycle_B = self.loss_lambda * self.criterionCycle(self.rec_B,self.real_B)
        # Identity loss
        self.loss_idt_A = self.loss_lambda * self.lambda_idt * self.criterionIdt(self.idt_A,self.real_B)
        self.loss_idt_B = self.loss_lambda * self.lambda_idt * self.criterionIdt(self.idt_B,self.real_A)
        # Total loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
    
    def backward_D(self):
        """ Loss for the discriminators
        """
        # D_A
        loss_D_A_real = self.criterionGAN(self.netD_A(self.real_B),True)
        loss_D_A_fake = self.criterionGAN(self.netD_A(self.fake_B.detach()),False)
        self.loss_D_A = (loss_D_A_real + loss_D_A_fake)*0.5
        self.loss_D_A.backward()
        # D_B
        loss_D_B_real = self.criterionGAN(self.netD_B(self.real_A),True)
        loss_D_B_fake = self.criterionGAN(self.netD_B(self.fake_A.detach()),False)
        self.loss_D_B = (loss_D_B_real + loss_D_B_fake)*0.5
        self.loss_D_B.backward()
    
    def optimize(self):
        """ Do forward pass and optimize parameters
        """
        # Forward pass
        self.forward()
        # G
        self.set_requires_grad([self.netD_A, self.netD_B],False) # First only G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.set_requires_grad([self.netD_A, self.netD_B],True) # Now D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.step += 1
        
    def load_model(self,detail_name=""):
        """ Load model weights from disc
            
            E.g. for model name test_model, loads the models
            checkpoints/test_model/models/nameG_A.pth etc.
        """
        for model in self.model_names:
            path = self.opt.checkpoints_dir +f"/{self.name}/models/{model}_{detail_name}.pth"
            getattr(self, 'net' + model).module.load_state_dict(torch.load(path))
    
    def save_model(self, detail_name="", **kwargs):
        """ Save the current models
            
            E.g. for model name test_model, saves the models to
            checkpoints/test_model/models/nameG_A.pth etc.
        """
        for model in self.model_names:
            path = self.opt.checkpoints_dir+f"/{self.name}/models/{model}_{detail_name}_{self.step:9d}.pth"
            latest_path = self.opt.checkpoints_dir+f"/{self.name}/models/{model}_{detail_name}_latest.pth"
            net = getattr(self, 'net' + model)
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.module.cpu().state_dict(), path)
                torch.save(net.module.cpu().state_dict(), latest_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), path)
                torch.save(net.cpu().state_dict(), latest_path)
        print("models saved!")
    
    def get_losses(self):
        """ Returns a dict containing the current losses
        """
        losses = dict()
        for loss in self.loss_names:
            losses[loss] = float(getattr(self, 'loss_' + loss).cpu().detach().numpy())
        wandb.log(losses, step=self.step)
        return losses
    
    def get_loss_string(self):
        """ Returns the latest losses merged into one string
        """
        losses = self.get_losses()
        loss_string = ""
        for loss_name, loss in losses.items():
            loss_string += loss_name + f": {loss:.3f}\t"
        lr =  self.optimizers[0].param_groups[0]['lr']
        loss_string += f"lr: {lr:.7f}\t"
        return loss_string
    
    def get_visuals(self):
        """ Returns a dict containing the current visuals
        """
        visuals = dict()
        for visual in self.visual_names:
            visuals[visual] = tensor_to_image(getattr(self, visual)) # .cpu().detach().permute(0,2,3,1).numpy()
        return visuals
    
    def save_visuals(self, **kwargs):
        """ Save recent visuals, i.e. real_A, fake_B etc., to disc
        """
        visuals = self.get_visuals()
        for visual, image in visuals.items():
            path = self.opt.checkpoints_dir +f"/{self.opt.name}/images/{visual}_{self.step}.png"
            wandb.log({visual: [wandb.Image(image, caption=visual)]}, step=self.step)
            save_image(path, image)
      
    def evaluate(self, list_of_images, **kwargs):
        """ evaluate test iamges and save multires grid
        """
        self.set_eval()

        with torch.no_grad():
            evaluated_images_A = []
            evaluated_images_B = []
            for image in tqdm(list_of_images):
                self.set_inputs(image)
                self.forward()
                evaluated_images_A.append(tensor_to_image(self.fake_A)[...,:3])
                evaluated_images_B.append(tensor_to_image(self.fake_B)[...,:3])
            path_A = self.opt.checkpoints_dir +f"/{self.opt.name}/eval_images/grid_fakeA_{self.step}.png"
            path_B = self.opt.checkpoints_dir +f"/{self.opt.name}/eval_images/grid_fakeB_{self.step}.png"
            image_A = util.draw_multires_figure(np.array(evaluated_images_A), n_columns=3)
            image_B = util.draw_multires_figure(np.array(evaluated_images_B), n_columns=3)
            wandb.log({'eval_fakeA': [wandb.Image(image_A)]}, step=self.step)
            wandb.log({'eval_fakeB': [wandb.Image(image_B)]}, step=self.step)
            save_image(path_A, np.array(image_A))
            save_image(path_B, np.array(image_B))

        self.set_train()
      
    
    def update_learning_rate(self):
        """ Update the learning rate at the end of each epoch
        """
        for scheduler in self.schedulers:
            scheduler.step()
        new_lr = self.optimizers[0].param_groups[0]['lr']
        wandb.log({'lr': new_lr}, step=self.step)
        #print("New lr: {:.6f}".format(new_lr))
    
    def set_requires_grad(self, nets, requires_grad=False):
        """ Set requies_grad = False for nets
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def print_logs(self, print_fn=print):
        # Print log
        log_string = self.get_loss_string()
        print_fn(log_string)

    def set_train(self):
        """ Set all nets to train
        """
        for model in self.model_names:
            getattr(self, "net"+model).train()
        
    def set_eval(self):
        """ Set all nets to train
        """
        for model in self.model_names:
            getattr(self, "net"+model).eval()
        
        #clauclate means, FID score etc.
        

class GeneratorTestModel():
    def __init__(self, opt):
        """ 
        Parameters:
        """
        
        # Stuff
        ensure_existance_paths_test(opt) # Create necessary directories for saving stuff
        
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids and torch.cuda.is_available() else torch.device('cpu')
        self.name = self.opt.name
        
        # Define model
        self.netG = networks.to_device(networks.ResNetGenerator(opt.n_input, opt.n_output, opt.forward_mask),self.gpu_ids)
        self.netG.eval()
        self.load_model(opt.load_model_name)

    def set_inputs(self,inputs):
        """ Set input images from dict inputs
        Parameters:
            
        """
        self.real = inputs.to(self.device)
    
    def forward(self):
        """ Perform forward pass
        """
        self.fake = self.netG(self.real)
    
    def load_model(self,detail_name=""):
        """ Load model weights from disc
            
            E.g. for model name test_model, loads the models
            checkpoints/test_model/models/nameG_A.pth etc.
        """
        path = self.opt.checkpoints_dir +f"/{self.name}/models/model_{detail_name}.pth"
        self.netG.module.load_state_dict(torch.load(path))
    
    def save_generated(self, data):
        """ Generate images from batch and save them
        """
        self.set_inputs(data["A"])
        self.forward()
        path = self.opt.checkpoints_dir +f"/{self.opt.name}/generated/{data['A_name']}_fake.png"
        save_image(path,tensor_to_image(self.fake)[...,:3])
        
