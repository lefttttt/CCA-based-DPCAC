import torch.nn as nn
import MinkowskiEngine as ME
from models.entropy_models.entropy_model import ScaleHyperprior
from models.network.module import *

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class Dynamic_PCAC(nn.Module):
    def __init__(self, chann=128, chann_mul=1, chann_div=1):
        super(Dynamic_PCAC, self).__init__()
        self.chann = chann
        self.chann_mul = chann_mul
        self.chann_div = chann_div
        self.encoder_main_net = nn.Sequential(
            Downsampling(3, chann, 2), 
            RCAB_group(chann),
            Downsampling(chann, chann, 2),
            RCAB_group(chann),
            nn.Sequential(
                DNConv(chann, chann, kernel_size=3, stride=1, dimension=3),
                ME.MinkowskiPReLU(),
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
                ME.MinkowskiPReLU(),
                DNConv(chann, 3, kernel_size=3, stride=1, dimension=3)
            )
        )
        self.position_module = GSM(3, 24)
        self.cross_attention = CrossAttn(chann, K_num=8)
        self.feature_transform_enc = RTM(chann,chann_mul,chann_div,type='enc')
        self.feature_transform_dec = RTM(chann,chann_mul,chann_div,type='dec')
        self.entropy_module = ScaleHyperprior(chann*chann_mul//chann_div)

    def generate_dec_ref(self, ref_Input, ref_ref_Input, ref_Coor, ref_ref_Coor):
        latent_ref = self.encoder_main_net(ref_Input)
        Coor_hat_ref = self.position_module(ref_Coor)
        latent_ref_ref = self.encoder_main_net(ref_ref_Input)
        Coor_hat_ref_ref = self.position_module(ref_ref_Coor)

        latent_pred_ref = self.cross_attention(Coor_hat_ref, Coor_hat_ref_ref, latent_ref_ref)

        latent_res_ref = ME.SparseTensor(
            latent_ref.F - latent_pred_ref.F,
            coordinate_map_key=latent_ref.coordinate_map_key,
            coordinate_manager=latent_ref.coordinate_manager,
            device=latent_ref.device
        )
        latent_res_ref = self.feature_transform_enc(latent_res_ref)

        latent_f_ref = RoundSTE.apply(latent_res_ref.F)

        latent_res_rec_ref = ME.SparseTensor(
            latent_f_ref,
            coordinate_map_key=latent_res_ref.coordinate_map_key,
            coordinate_manager=latent_res_ref.coordinate_manager,
            device=latent_res_ref.device
        )
        latent_res_rec_ref = self.feature_transform_dec(latent_res_rec_ref)
        latent_rec_ref = ME.SparseTensor(
            (latent_res_rec_ref.F + latent_pred_ref.F),
            coordinate_map_key=latent_res_rec_ref.coordinate_map_key,
            coordinate_manager=latent_res_rec_ref.coordinate_manager,
            device=latent_res_rec_ref.device
        )
        Output_ref = self.decoder_main_net(latent_rec_ref)
        return Output_ref

    def forward(self, Input, ref_Input, ref_ref_Input, Coor, ref_Coor, ref_ref_Coor, external_Output_ref=None):
        if external_Output_ref is None:
            with torch.no_grad():
                Output_ref = self.generate_dec_ref(ref_Input, ref_ref_Input, ref_Coor, ref_ref_Coor)
        else:
            Output_ref = external_Output_ref 

        # Encoder
        # Use Output_ref.detach() to block the gradient and simulate the test phase
        latent_ref = self.encoder_main_net(Output_ref.detach()) #
        Coor_hat_ref = self.position_module(ref_Coor) 
        latent = self.encoder_main_net(Input)
        Coor_hat = self.position_module(Coor)
        latent_pred = self.cross_attention(Coor_hat, Coor_hat_ref, latent_ref)
        latent_res = ME.SparseTensor(latent.F - latent_pred.F, coordinate_map_key=latent.coordinate_map_key,
                                     coordinate_manager=latent.coordinate_manager, device=latent.device)
        latent_res = self.feature_transform_enc(latent_res)

        # Entropy Model
        latent_f, latent_likelihoods, hyper_likelihoods = self.entropy_module(latent_res, Coor_hat)

        # Decoder
        latent_res_rec = ME.SparseTensor(latent_f, coordinate_map_key=latent_res.coordinate_map_key,
                                     coordinate_manager=latent_res.coordinate_manager, device=latent_res.device)
        latent_res_rec = self.feature_transform_dec(latent_res_rec)
        latent_rec = ME.SparseTensor((latent_res_rec.F + latent_pred.F), coordinate_map_key=latent_res.coordinate_map_key,
                                     coordinate_manager=latent_res.coordinate_manager, device=latent_res.device)
        Output = self.decoder_main_net(latent_rec)

        return Output, latent_likelihoods, hyper_likelihoods
    
    def compress(self, Input, Coor, local_ref_cache):
        ref_Input = ME.SparseTensor(
            features=local_ref_cache['feats'].to(Input.device),
            coordinates=local_ref_cache['coords'].to(Input.device).int(),
            device=Input.device
            )
        ref_Coor = ME.SparseTensor(
            features=local_ref_cache['coords'].to(Coor.device)[:, 1:].float() / 1024.0,
            coordinate_map_key=ref_Input.coordinate_map_key,
            coordinate_manager=ref_Input.coordinate_manager,
            device=Input.device
        )

        # Cross-Coordinate Attention
        ref_latent = self.encoder_main_net(ref_Input)
        ref_Coor_hat = self.position_module(ref_Coor)
        ref_latent_feat = ref_latent.F
        ref_Coor_hat_coord = ref_Coor_hat.C[:, 1:]
        ref_Coor_hat_feat = ref_Coor_hat.F
        
        Coor_hat = self.position_module(Coor)
        latent_pred = self.cross_attention.test_forward(Coor_hat, ref_latent_feat, ref_Coor_hat_coord, ref_Coor_hat_feat)

        # Encoder
        latent = self.encoder_main_net(Input)
        latent_res = ME.SparseTensor(latent.F - latent_pred.F, coordinate_map_key=latent.coordinate_map_key,
                                     coordinate_manager=latent.coordinate_manager, device=latent.device)
        latent_res = self.feature_transform_enc(latent_res)
        
        # Entropy Encoder
        latent_str1, latent_str2, hyper_str = self.entropy_module.compress(latent_res, Coor_hat)
        return latent_str1, latent_str2, hyper_str
    
    def decompress(self, latent_str1, latent_str2, hyper_str, Coor, local_ref_cache):
        
        ref_Input = ME.SparseTensor(
            features=local_ref_cache['feats'].to(Coor.device),
            coordinates=local_ref_cache['coords'].to(Coor.device).int(),
            device=Coor.device
            )
        ref_Coor = ME.SparseTensor(
            features=local_ref_cache['coords'].to(Coor.device)[:, 1:].float() / 1024.0,
            coordinate_map_key=ref_Input.coordinate_map_key,
            coordinate_manager=ref_Input.coordinate_manager,
            device=Coor.device
        )
        Coor_hat = self.position_module(Coor)

        # Cross-Coordinate Attention
        ref_latent = self.encoder_main_net(ref_Input)
        ref_Coor_hat = self.position_module(ref_Coor)
        ref_latent_feat = ref_latent.F
        ref_Coor_hat_coord = ref_Coor_hat.C[:, 1:]
        ref_Coor_hat_feat = ref_Coor_hat.F
        latent_pred = self.cross_attention.test_forward(Coor_hat, ref_latent_feat, ref_Coor_hat_coord, ref_Coor_hat_feat)

        # Entropy Decoder
        latent_f = self.entropy_module.decompress(latent_str1, latent_str2, hyper_str, Coor_hat)

        # Decoder
        latent_res_rec = ME.SparseTensor(latent_f, coordinate_map_key=Coor_hat.coordinate_map_key,
                                        coordinate_manager=Coor_hat.coordinate_manager, device=Coor_hat.device)
        latent_res_rec = self.feature_transform_dec(latent_res_rec)
        latent_rec = ME.SparseTensor((latent_res_rec.F + latent_pred.F), coordinate_map_key=latent_pred.coordinate_map_key,
                                  coordinate_manager=latent_pred.coordinate_manager, device=latent_pred.device)
        Output = self.decoder_main_net(latent_rec)

        return Output
