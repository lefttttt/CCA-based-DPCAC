import math
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.google import get_scale_table
import MinkowskiEngine as ME
from models.utils.ConvLayer4 import *
from models.KNN_CUDA.knn_cuda import KNN

class CrossAttn(nn.Module):
    def __init__(self,chann, K_num=8):
        super(CrossAttn, self).__init__()
        self.K_num = K_num
        self.knn = KNN(k=K_num, transpose_mode=True)
        self.W_v =nn.Conv1d(chann,chann,1)
        self.W_o=nn.Conv1d(chann,chann,1)
        self.W_out = ME.MinkowskiConvolution(chann , chann*2 , kernel_size=1, stride=1, dimension=3)
        # self.coor_conv = nn.Sequential(
        #     ME.MinkowskiConvolution(3, 16, kernel_size=3, stride=2, dimension=3),
        #     ME.MinkowskiBatchNorm(16),
        #     ME.MinkowskiReLU(),
        #     ME.MinkowskiConvolution(16, 16, kernel_size=3, stride=2, dimension=3),
        #     ME.MinkowskiBatchNorm(16),
        #     ME.MinkowskiReLU(),
        #     ME.MinkowskiConvolution(16, 16, kernel_size=3, stride=2, dimension=3)
        # )

    def forward(self, spinput, coor_hat, mask):
        list_of_coor, _ = spinput.decomposed_coordinates_and_features
        batch_size = len(spinput.decomposed_coordinates_and_features[0])
        output = []


        for i in range(batch_size):
            with torch.no_grad():
                xyz = coor_hat.coordinates_at(i).clone().float()
                xyz_ref = xyz[mask.features_at(i)[:, 0] == 1]
                xyz_pred = xyz[mask.features_at(i)[:, 0] == 0]
                dist, idx = self.knn(xyz_ref.unsqueeze(0), xyz_pred.unsqueeze(0))
                idx = idx.squeeze(0)

            q = coor_hat.features_at(i)[mask.features_at(i)[:, 0] == 0]
            k_raw = coor_hat.features_at(i)[mask.features_at(i)[:, 0] == 1]

            k = k_raw[idx]
            q = q.unsqueeze(1)

            v_raw = spinput.features_at(i)[mask.features_at(i)[:, 0] == 1]
            v_raw = self.W_v(v_raw.transpose(0, 1).unsqueeze(0))
            v_raw = v_raw.squeeze(0).transpose(0, 1)
            v = v_raw[idx]

            attn_score = q @ k.transpose(1, 2)
            attn_score = (attn_score) / math.sqrt(q.shape[-1])
            attn_score = torch.nn.functional.softmax(attn_score, dim=-1)
            out = attn_score @ v
            out = out.squeeze(1)
            out =self.W_o(out.transpose(0, 1).unsqueeze(0))
            out = out.squeeze(0).transpose(0, 1)
            out_all = torch.zeros(spinput.features_at(i).shape).to(spinput.device)
            # out_all[mask.features_at(i)[:, 0] == 0] += out
            out_all[mask.features_at(i)[:, 0] == 0] += out
            output.append(out_all)
            # coor.append(spinput.coordinates_at(i)[mask.features_at(i)[:, 0] == 0])
            # feat.append(out)

        output = torch.cat(output, dim=0)
        spout = ME.SparseTensor(output, coordinate_map_key=spinput.coordinate_map_key,
                                           coordinate_manager=spinput.coordinate_manager, device=spinput.device)
        # print(spinput.F[:10, :])
        # print(spout.F[:10, :])
        spoutput = self.W_out(spout)
        return spoutput

class ScaleHyperpriorHyperAnalysis(nn.Module):
    def __init__(self,chann):
        super().__init__()
        self.model = nn.Sequential(
            PatialConv(chann,chann,kernel_size=3,stride=1,dimension=3),
            ME.MinkowskiReLU(),
            PatialConv(chann, chann, kernel_size=3, stride=1, dimension=3),
            PatialConv(chann, chann, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiReLU(),
            PatialConv(chann, chann, kernel_size=3, stride=1, dimension=3),
            PatialConv(chann, chann, kernel_size=3, stride=2, dimension=3),
        )

    def forward(self,x):
        return self.model(x)


class ScaleHyperpriorHyperSynthesis(nn.Module):
    def __init__(self,chann):
        super().__init__()
        self.model = nn.Sequential(
            PatialConvTr(chann, chann, kernel_size=3, stride=2, dimension=3),
            PatialConv(chann, chann, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiReLU(),
            PatialConvTr(chann, chann*2, kernel_size=3, stride=2, dimension=3),
            PatialConv(chann*2, chann*2, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiReLU(),
            PatialConv(chann*2, chann*2, kernel_size=3, stride=1, dimension=3),
        )
    def forward(self,x):
        return self.model(x)


class ScaleHyperprior(nn.Module):
    def __init__(self,network_channels):
        super().__init__()
        self.channels = network_channels
        self.hyper_bottleneck = EntropyBottleneck(channels=network_channels)
        self.latent_bottleneck = GaussianConditional(scale_table=None)

        self.latent_analysis = ScaleHyperpriorHyperAnalysis(network_channels)
        self.latent_synthesis = ScaleHyperpriorHyperSynthesis(network_channels)
        self.corss_attn = CrossAttn(network_channels, K_num=8)
        self.entropy_paramters = nn.Sequential(
            ME.MinkowskiConvolution(network_channels*4, network_channels*3, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(network_channels*3, network_channels*2, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(network_channels*2, network_channels*2, kernel_size=1, stride=1, dimension=3),
        )


    def generate_mask(self,input_sp):
        with torch.no_grad():
            list_of_coords, list_of_features = input_sp.decomposed_coordinates_and_features
            mask_all = []
            mask_inv_all = []
            for batch in range(len(list_of_coords)):
                f = input_sp.features_at(batch)
                mask = torch.zeros(f.shape[0], f.shape[1]//2).to(f.device)
                mask[0::2] = 0
                mask[1::2] = 1

                mask_inv = torch.zeros(f.shape[0], f.shape[1]//2).to(f.device)
                mask_inv[0::2] = 1
                mask_inv[1::2] = 0
                mask_all.append(mask)
                mask_inv_all.append(mask_inv)
            mask_all = torch.cat(mask_all,dim=0)
            mask_sp = ME.SparseTensor(features=mask_all, coordinate_map_key=input_sp.coordinate_map_key,
                                      coordinate_manager=input_sp.coordinate_manager, device=input_sp.device)
            mask_inv_all = torch.cat(mask_inv_all, dim=0)
            mask_inv_sp = ME.SparseTensor(features=mask_inv_all, coordinate_map_key=input_sp.coordinate_map_key,
                                          coordinate_manager=input_sp.coordinate_manager, device=input_sp.device)

        return mask_sp,mask_inv_sp

    def forward(self, input_sp, coor_hat):
        hyper_latent = self.latent_analysis(input_sp)
        hyper_latent_f = hyper_latent.F.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # [1,C,N]

        noisy_hyper_latent, hyper_latent_likelihoods = self.hyper_bottleneck(hyper_latent_f)
        noisy_hyper_latent = noisy_hyper_latent.squeeze(-1).squeeze(0).transpose(0, 1)

        hyper_latent_noisy = ME.SparseTensor(noisy_hyper_latent, coordinate_map_key=hyper_latent.coordinate_map_key,
                                             coordinate_manager=hyper_latent.coordinate_manager, device=input_sp.device)
        scales_mean = self.latent_synthesis(hyper_latent_noisy)

        means_hat, scales_hat = scales_mean.F.chunk(2, 1)
        input_f_hat = input_sp.F.clone()
        with torch.no_grad():
            mask_sp, mask_inv_sp = self.generate_mask(scales_mean)

        input_f_hat = self.latent_bottleneck.quantize(input_f_hat, means=means_hat,
                                                      mode="noise" if self.training else "dequantize")

        input_sp_hat = ME.SparseTensor(input_f_hat, coordinate_map_key=input_sp.coordinate_map_key,
                                       coordinate_manager=input_sp.coordinate_manager, device=input_sp.device)
        input_sp_hat = input_sp_hat * mask_sp
        context_pred = self.corss_attn(input_sp_hat, coor_hat, mask_sp)

        mask_inv_sp = ME.cat(mask_inv_sp,mask_inv_sp)
        context_pred = context_pred * mask_inv_sp


        ctx_params = ME.cat(scales_mean, context_pred)
        ctx_params = self.entropy_paramters(ctx_params)
        mask_sp = ME.cat(mask_sp, mask_sp)
        ctx_params = ctx_params * mask_inv_sp + scales_mean * mask_sp

        input_latent = input_sp.F.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        feature_sm_mean_f, feature_sm_scale_f = ctx_params.F.chunk(2, 1)
        feature_sm_mean_f = feature_sm_mean_f.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        feature_sm_scale_f = feature_sm_scale_f.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        input_f_noise, latent_likelihoods = self.latent_bottleneck(input_latent, feature_sm_scale_f,
                                                                   means=feature_sm_mean_f)
        input_f_noise = input_f_noise.squeeze(0).squeeze(-1).transpose(0, 1)
        return input_f_noise, latent_likelihoods, hyper_latent_likelihoods

    def update(self, force=False):
        hyper_bottleneck_updated = self.hyper_bottleneck.update(force=force)  # type: ignore
        latent_bottleneck_updated = self.latent_bottleneck.update_scale_table(
            get_scale_table(), force=force
        )
        return hyper_bottleneck_updated, latent_bottleneck_updated

    def compress(self, input_sp, coor_latent):
        hyper_latent = self.latent_analysis(input_sp)
        hyper_latent_f = hyper_latent.F.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        hyper_strings = self.hyper_bottleneck.compress(hyper_latent_f)
        noisy_hyper_latent_decode = self.hyper_bottleneck.decompress(hyper_strings, hyper_latent_f.shape[2:])
        noisy_hyper_latent = noisy_hyper_latent_decode.squeeze(0).squeeze(-1).transpose(0, 1)
        hyper_latent_noisy = ME.SparseTensor(noisy_hyper_latent, coordinate_map_key=hyper_latent.coordinate_map_key,
                                             coordinate_manager=hyper_latent.coordinate_manager, device=input_sp.device)
        scales_mean = self.latent_synthesis(hyper_latent_noisy)

        input_f_hat = torch.zeros(input_sp.F.shape).to(input_sp.device)
        mask_sp, mask_inv_sp = self.generate_mask(scales_mean)

        input_latent = input_sp.F[mask_sp.F[:, 0] == 1].transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        feature_sm_mean_f, feature_sm_scale_f = scales_mean.F.chunk(2, 1)
        feature_sm_scale_f = feature_sm_scale_f[mask_sp.F[:, 0] == 1].transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        feature_sm_mean_f = feature_sm_mean_f[mask_sp.F[:, 0] == 1].transpose(0, 1).unsqueeze(0).unsqueeze(-1)

        indexes1 = self.latent_bottleneck.build_indexes(feature_sm_scale_f)
        latent_strings1 = self.latent_bottleneck.compress(input_latent, indexes1, means=feature_sm_mean_f)
        latent_decode1 = self.latent_bottleneck.decompress(latent_strings1, indexes1, means=feature_sm_mean_f)

        latent_decode1 = latent_decode1.squeeze(0).squeeze(-1).transpose(0, 1)
        input_f_hat[mask_sp.F[:, 0] == 1] = latent_decode1
        input_sp_hat = ME.SparseTensor(input_f_hat, coordinate_map_key=input_sp.coordinate_map_key,
                                       coordinate_manager=input_sp.coordinate_manager, device=input_sp.device)
        context_pred = self.corss_attn(input_sp_hat, coor_latent, mask_sp)

        ctx_params = ME.cat(scales_mean, context_pred)
        ctx_params = self.entropy_paramters(ctx_params)

        input_latent = input_sp.F[mask_sp.F[:, 0] == 0].transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        feature_sm_mean_f, feature_sm_scale_f = ctx_params.F.chunk(2, 1)
        feature_sm_scale_f = feature_sm_scale_f[mask_sp.F[:, 0] == 0].transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        feature_sm_mean_f = feature_sm_mean_f[mask_sp.F[:, 0] == 0].transpose(0, 1).unsqueeze(0).unsqueeze(-1)

        indexes2 = self.latent_bottleneck.build_indexes(feature_sm_scale_f)
        latent_strings2 = self.latent_bottleneck.compress(input_latent, indexes2, means=feature_sm_mean_f)

        return latent_strings1, latent_strings2, hyper_strings

    def decompress(self, latent_strings1, latent_strings2, hyper_strings, coor_latent):
        hyper_coord_flag_f = torch.zeros((coor_latent.F.shape[0], self.channels)).to(coor_latent.device)
        hyper_coord_flag = ME.SparseTensor(features=hyper_coord_flag_f, coordinate_map_key=coor_latent.coordinate_map_key,
                                       coordinate_manager=coor_latent.coordinate_manager, device=coor_latent.device)
        hyper_latent_coord_flag = self.latent_analysis(hyper_coord_flag)

        noisy_hyper_latent_decode = self.hyper_bottleneck.decompress(hyper_strings, (hyper_latent_coord_flag.F.shape[0], 1))

        noisy_hyper_latent = noisy_hyper_latent_decode.squeeze(0).squeeze(-1).transpose(0, 1)
        hyper_latent_noisy = ME.SparseTensor(noisy_hyper_latent, coordinate_map_key=hyper_latent_coord_flag.coordinate_map_key,
                                             coordinate_manager=hyper_latent_coord_flag.coordinate_manager,
                                             device=hyper_latent_coord_flag.device)
        scales_mean = self.latent_synthesis(hyper_latent_noisy)

        input_f_hat = torch.zeros(scales_mean.F.shape[0], scales_mean.F.shape[1] // 2).to(scales_mean.device)
        mask_sp, mask_inv_sp = self.generate_mask(scales_mean)

        feature_sm_mean_f, feature_sm_scale_f = scales_mean.F.chunk(2, 1)
        feature_sm_scale_f = feature_sm_scale_f[mask_sp.F[:, 0] == 1].transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        feature_sm_mean_f = feature_sm_mean_f[mask_sp.F[:, 0] == 1].transpose(0, 1).unsqueeze(0).unsqueeze(-1)

        indexes1 = self.latent_bottleneck.build_indexes(feature_sm_scale_f)
        latent_decode1 = self.latent_bottleneck.decompress(latent_strings1, indexes1, means=feature_sm_mean_f)

        latent_decode1 = latent_decode1.squeeze(0).squeeze(-1).transpose(0, 1)
        input_f_hat[mask_sp.F[:, 0] == 1] = latent_decode1
        input_sp_hat = ME.SparseTensor(input_f_hat, coordinate_map_key=coor_latent.coordinate_map_key,
                                       coordinate_manager=coor_latent.coordinate_manager, device=coor_latent.device)
        context_pred = self.corss_attn(input_sp_hat, coor_latent, mask_sp)

        ctx_params = ME.cat(scales_mean, context_pred)
        ctx_params = self.entropy_paramters(ctx_params)

        feature_sm_mean_f, feature_sm_scale_f = ctx_params.F.chunk(2, 1)
        feature_sm_scale_f = feature_sm_scale_f[mask_sp.F[:, 0] == 0].transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        feature_sm_mean_f = feature_sm_mean_f[mask_sp.F[:, 0] == 0].transpose(0, 1).unsqueeze(0).unsqueeze(-1)

        indexes2 = self.latent_bottleneck.build_indexes(feature_sm_scale_f)
        latent_decode2 = self.latent_bottleneck.decompress(latent_strings2, indexes2, means=feature_sm_mean_f)
        latent_decode2 = latent_decode2.squeeze(0).squeeze(-1).transpose(0, 1)
        input_f_hat[mask_sp.F[:, 0] == 0] = latent_decode2

        return input_f_hat