import torch.nn as nn
import MinkowskiEngine as ME
from models.entropy_models.entropy_model import ScaleHyperprior
import torch
from models.network.module import *


class Static_PCAC(nn.Module):
    def __init__(self, chann=96):
        super(Static_PCAC, self).__init__()
        self.encoder_main_net = nn.Sequential(
            Downsampling(3, chann, 2), 
            RCAB_group(chann),
            Downsampling(chann, chann, 2),
            RCAB_group(chann),
            nn.Sequential(
                DNConv(chann, chann, kernel_size=3, stride=1, dimension=3),
                DNConv(chann, chann, kernel_size=3, stride=2, dimension=3),
            )
        )
        self.decoder_main_net = nn.Sequential(
            Upsampling(chann, chann, 2),
            RCAB_group(chann),
            Upsampling(chann, chann, 2),
            RCAB_group(chann),
            nn.Sequential(
                DNConvTr(chann, chann, kernel_size=3, stride=2, dimension=3),
                DNConv(chann, 3, kernel_size=3, stride=1, dimension=3)
            )
        )
        self.position_module = GSM(3, 16)
        self.density_module_encoder = DensityModule(chann)
        self.density_module_decoder = DensityModule(chann)
        self.entropy_module = ScaleHyperprior(chann)


    def forward(self, Input, Coor, Density):
        # Encoder
        latent = self.encoder_main_net(Input)
        Density_enc = self.density_module_encoder(Density)
        latent = ME.SparseTensor(latent.F * Density_enc.F, coordinate_map_key=latent.coordinate_map_key,
                                 coordinate_manager=latent.coordinate_manager, device=latent.device)
        
        # Entropy Model
        Coor_hat = self.position_module(Coor)
        latent_f, latent_likelihoods, hyper_likelihoods = self.entropy_module(latent, Coor_hat)

        # Decoder
        Density_dec = self.density_module_decoder(Density)
        Output = ME.SparseTensor(latent_f * Density_dec.F, coordinate_map_key=latent.coordinate_map_key,
                                  coordinate_manager=latent.coordinate_manager, device=latent.device)
        Output = self.decoder_main_net(Output)

        return Output, latent_likelihoods, hyper_likelihoods
    
    def compress(self, Input, Coor, Density):
        # Encoder
        latent = self.encoder_main_net(Input)
        Density_enc = self.density_module_encoder(Density)
        latent = ME.SparseTensor(latent.F * Density_enc.F, coordinate_map_key=latent.coordinate_map_key,
                                 coordinate_manager=latent.coordinate_manager, device=latent.device)

        # Entropy Encoder
        Coor_hat = self.position_module(Coor)
        latent_str1, latent_str2, hyper_str = self.entropy_module.compress(latent, Coor_hat)

        return latent_str1, latent_str2, hyper_str
    
    def decompress(self, latent_str1, latent_str2, hyper_str, Coor, Density):
        # Entropy Decoder
        Coor_hat = self.position_module(Coor)
        Density_dec = self.density_module_decoder(Density)
        latent_f = self.entropy_module.decompress(latent_str1, latent_str2, hyper_str, Coor_hat)

        # Decoder
        latent = ME.SparseTensor(latent_f * Density_dec.F, coordinate_map_key=Density_dec.coordinate_map_key,
                                  coordinate_manager=Density_dec.coordinate_manager, device=Density_dec.device)
        Output = self.decoder_main_net(latent)

        return Output
