import torch
import torch.nn as nn
import itertools
import networks


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
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.lambda_idt = 0.5
        self.loss_lambda = 10.0
        
        # Define models
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        self.netG_A = networks.to_device(networks.ResNetGenerator(opt.n_input, opt.n_output),self.gpu_ids)
        self.netG_B = networks.to_device(networks.ResNetGenerator(opt.n_input, opt.n_output),self.gpu_ids)
        self.netD_A = networks.to_device(networks.NLayerDiscriminator(opt.n_output),self.gpu_ids)
        self.netD_B = networks.to_device(networks.NLayerDiscriminator(opt.n_output),self.gpu_ids)
        
        # Define losses
        self.criterionGAN = networks.GANLoss().to(self.device)
        self.criterionCycle = nn.L1Loss().to(self.device)
        self.criterionIdt = nn.L1Loss().to(self.device)
        
        # Define optimizers
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_D)
        
        # Define schedulers
        self.schedulers = [networks.get_optimizer_scheduler(optimizer,opt) for optimizer in self.optimizers]
        
    
    
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
        self.loss_cycle_A = self.criterionCycle(self.rec_A,self.real_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B,self.real_B)
        # Identity loss
        self.loss_idt_A = self.criterionIdt(self.idt_A,self.real_B)
        self.loss_idt_B = self.criterionIdt(self.idt_B,self.real_A)
        # Total loss
        self.loss_G = self.loss_G_A + self.loss_G_B
        self.loss_G += self.loss_lambda*(self.loss_cycle_A + self.loss_cycle_B)
        self.loss_G += self.loss_lambda*self.lambda_idt*(self.loss_idt_A + self.loss_idt_B)
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
        
    
    def update_learning_rate(self):
        """ Update the learning rate at the end of each epoch
        """
        for scheduler in self.schedulers:
            scheduler.step()
        new_lr = self.optimizers[0].param_groups[0]['lr']
        print("New lr: {:.6f}".format(new_lr))
    
    def set_requires_grad(self, nets, requires_grad=False):
        """ Set requies_grad = False for nets
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad