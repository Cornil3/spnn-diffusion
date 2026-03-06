import torch
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
import os
from clip.models.ko_clip import KoCLIPWrapper

from diffusion_model.models.diffusion_model import DiffusionModel
from diffusion_model.network.unet_wrapper import UnetWrapper
from diffusion_model.network.unet import Unet
from helper.painter import Painter
from helper.cond_encoder import ClassEncoder
from helper.cond_encoder import CLIPEncoder
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.sampler.ddim import DDIM

IMAGE_SHAPE = (3, 32, 32)
CONFIG_PATH = './configs/cifar10_config.yaml'

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'using device : {device}\t' + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU' ))
    
    painter = Painter()
    
    sampler = DDIM(CONFIG_PATH)
    vae = VariationalAutoEncoder(CONFIG_PATH)
    #trainer = Trainer(vae, loss_fn = vae.loss)
    #trainer.train(data_loader, 1000, VAE_FILE_NAME, True)
    
    clip = KoCLIPWrapper()
    cond_encoder = CLIPEncoder(clip, CONFIG_PATH)
    #cond_encoder = ConditionEncoder(CONFIG_PATH)
    network = UnetWrapper(Unet, CONFIG_PATH, cond_encoder)
    dm = LatentDiffusionModel(network, sampler, vae)
    painter = Painter()
    #data_generator = DataGenerator()
    #data_loader = data_generator.cifar10(path = './datasets' ,batch_size = 32)
    
    #dm = DiffusionModel(network, sampler, IMAGE_SHAPE)
    
    #sample = dm(2)
    sample = dm.sample(2, y = '..', gamma = 2)
    painter.show_images(sample)
    
    