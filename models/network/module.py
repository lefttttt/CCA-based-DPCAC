import torch.nn as nn
import MinkowskiEngine as ME
import torch
import math
from models.KNN_CUDA.knn_cuda import KNN

class DNConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,dimension=3,bias=False):
        super().__init__()
        self.input_conv = ME.MinkowskiConvolution(in_channels,out_channels,kernel_size,stride,dimension=dimension,bias=bias)
        self.mask_conv = ME.MinkowskiConvolution(1, 1, kernel_size, stride, dimension=3, bias=False)

        torch.nn.init.constant_(self.mask_conv.kernel.data,1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False
        #alpha = kernel_size*kernel_size*kernel_size/2
        #self.alpha = torch.nn.Parameter(torch.ones(size=[1])*alpha, requires_grad=True)

    def forward(self,input,mask_sp=None):
        output = self.input_conv(input)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.expand_as(output.F)
        else:
            output_bias =torch.zeros_like(output.F).to(input.device)
        with torch.no_grad():
            if mask_sp==None:
                mask_sp = ME.SparseTensor(torch.ones((input.F.shape[0],1)), coordinate_map_key=input.coordinate_map_key,
                                          coordinate_manager=input.coordinate_manager, device=input.device)
            else:
                mask_sp = ME.SparseTensor(mask_sp.F[:,:1],
                                          coordinate_map_key=input.coordinate_map_key,
                                          coordinate_manager=input.coordinate_manager, device=input.device)
            output_mask = self.mask_conv(mask_sp)

            output_mask_f= output_mask.F
            output_mask_f_zero = torch.ones_like(output_mask_f)
            output_mask_f_zero[output_mask_f == 0] = 0

            output_mask_f = output_mask_f.expand_as(output.F)
            output_mask_f_zero = output_mask_f_zero.expand_as(output.F)
            output_mask_f = torch.clamp(output_mask_f, min=1)
        output_f = (output.F-output_bias)*9*output_mask_f_zero/output_mask_f+output_bias
        #output_f = (output.F-output_bias)*self.alpha*output_mask_f_zero/output_mask_f+output_bias
        #print(self.alpha)
        output = ME.SparseTensor(output_f, coordinate_map_key=output.coordinate_map_key,
                                 coordinate_manager=output.coordinate_manager, device=input.device)
        return output


class DNConvTr(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,dimension=3,bias=False):
        super().__init__()
        self.input_conv = ME.MinkowskiConvolutionTranspose(in_channels,out_channels,kernel_size,stride,dimension=dimension,bias=bias)

        self.mask_conv = ME.MinkowskiConvolutionTranspose(1, 1, kernel_size, stride, dimension=3, bias=False)

        torch.nn.init.constant_(self.mask_conv.kernel.data,1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        self.stride = stride

        #alpha = kernel_size * kernel_size * kernel_size / 2
        #self.alpha = torch.nn.Parameter(torch.ones(size=[1])*alpha, requires_grad=True)

    def forward(self,input,mask_sp=None):
        output = self.input_conv(input)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.expand_as(output.F)
        else:
            output_bias =torch.zeros_like(output.F).to(input.device)
        with torch.no_grad():
            if mask_sp==None:
                mask_sp = ME.SparseTensor(torch.ones((input.F.shape[0],1)), coordinate_map_key=input.coordinate_map_key,
                                          coordinate_manager=input.coordinate_manager, device=input.device)
            else:
                mask_sp = ME.SparseTensor(mask_sp.F[:,:1],
                                          coordinate_map_key=input.coordinate_map_key,
                                          coordinate_manager=input.coordinate_manager, device=input.device)
            output_mask = self.mask_conv(mask_sp)
            output_mask_f = output_mask.F
            output_mask_f_zero = torch.ones_like(output_mask_f)
            output_mask_f_zero[output_mask_f == 0] = 0

            output_mask_f = output_mask_f.expand_as(output.F)
            output_mask_f_zero = output_mask_f_zero.expand_as(output.F)
            output_mask_f = torch.clamp(output_mask_f, min=1)
        output_f = (output.F - output_bias)*9*output_mask_f_zero / output_mask_f + output_bias
        #output_f = (output.F - output_bias)*self.alpha*output_mask_f_zero / output_mask_f + output_bias

        output = ME.SparseTensor(output_f, coordinate_map_key=output.coordinate_map_key,
                                 coordinate_manager=output.coordinate_manager, device=input.device)

        return output


class Downsampling(nn.Module):
    def __init__(self,in_chann,out_chann,stride=1,kernel_size=3):
        nn.Module.__init__(self)

        self.path = nn.Sequential(
            DNConv(in_chann, out_chann, kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiPReLU(),
            DNConv(out_chann, out_chann, kernel_size=kernel_size, stride=1, dimension=3),
            ME.MinkowskiPReLU()
        )
        self.downsample = nn.Sequential(
            ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiConvolution(in_chann, out_chann, kernel_size=1, stride=1, dimension=3)
        )

    def forward(self,x):
        out = self.path(x)
        identity = self.downsample(x)
        out = out+identity
        return out


class Upsampling(nn.Module):
    def __init__(self,in_chann,out_chann,stride=1,kernel_size=3):
        nn.Module.__init__(self)
        self.path = nn.Sequential(
            DNConvTr(in_chann, out_chann, kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiPReLU(),
            DNConv(out_chann, out_chann, kernel_size=kernel_size, stride=1, dimension=3),
            ME.MinkowskiPReLU()
        )
        self.upsample = nn.Sequential(
            ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiConvolution(in_chann, out_chann, kernel_size=1, stride=1, dimension=3),)


    def forward(self,x):
        out = self.path(x)
        identity = self.upsample(x)
        out = out+identity
        return out


class RCAB(nn.Module):
    def __init__(self, channel, kernel_size, reduction=8):
        super(RCAB,self).__init__()
        self.main_net = nn.Sequential(
            DNConv(channel, channel, kernel_size = kernel_size),
            ME.MinkowskiPReLU(),
            DNConv(channel, channel, kernel_size = kernel_size),
            ME.MinkowskiPReLU(),
        )
        self.CA = nn.Sequential(
            ME.MinkowskiGlobalAvgPooling(),
            ME.MinkowskiConvolution(channel, channel // reduction, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiPReLU(),
            ME.MinkowskiConvolution(channel // reduction, channel, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiSigmoid()
        )
        self.mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        out = self.main_net(x)
        CA_weight = self.CA(out)
        out = self.mul(out, CA_weight)
        out += x
        return out


class RCAB_group(nn.Module):
    def __init__(self,chann):
        super(RCAB_group,self).__init__()

        self.layer1 = RCAB(chann,3)
        self.layer2 = RCAB(chann, 3)

    def forward(self,x):
        res = self.layer1(x)
        res = self.layer2(res)
        res +=x
        return res
    

class CrossAttn(nn.Module):
    def __init__(self,chann, K_num=8):
        super(CrossAttn, self).__init__()
        self.K_num = K_num
        self.knn = KNN(k=K_num, transpose_mode=True)
        self.W_v =nn.Conv1d(chann,chann,1)
        self.W_o=nn.Conv1d(chann,chann,1)
        self.W_out = ME.MinkowskiConvolution(chann , chann , kernel_size=1, stride=1, dimension=3)

    def forward(self, Coor_hat, Coor_ref_hat, Feat_ref):
        batch_size = len(Feat_ref.decomposed_coordinates_and_features[0])
        pred_all = []

        for i in range(batch_size):
            with torch.no_grad():
                xyz_pred = Coor_hat.coordinates_at(i).clone().float()
                xyz_ref = Coor_ref_hat.coordinates_at(i).clone().float()

                _, idx = self.knn(xyz_ref.unsqueeze(0), xyz_pred.unsqueeze(0))
                idx = idx.squeeze(0)

            q = Coor_hat.features_at(i).clone().float()
            q = q.unsqueeze(1)
            k_raw = Coor_ref_hat.features_at(i).clone().float()
            k = k_raw[idx]

            v_raw = Feat_ref.features_at(i).clone().float()
            v_raw = self.W_v(v_raw.transpose(0, 1).unsqueeze(0))
            v_raw = v_raw.squeeze(0).transpose(0, 1)
            v = v_raw[idx]

            attn_score = q @ k.transpose(1, 2)
            attn_score = (attn_score) / math.sqrt(Coor_hat.features_at(i).shape[-1])
            attn_score = torch.nn.functional.softmax(attn_score, dim=-1)
            pred = attn_score @ v
            pred = pred.squeeze(1)
            pred =self.W_o(pred.transpose(0, 1).unsqueeze(0))
            pred = pred.squeeze(0).transpose(0, 1)
            pred_all.append(pred)

        Feat_pred = ME.SparseTensor(torch.cat(pred_all, dim=0), coordinate_map_key=Feat_ref.coordinate_map_key,
                                           coordinate_manager=Feat_ref.coordinate_manager, device=Feat_ref.device)
        Feat_pred = self.W_out(Feat_pred)
        return Feat_pred
    
    def test_forward(self, Coor_hat, ref_latent_feat, ref_Coor_hat_coord, ref_Coor_hat_feat):
        xyz_pred = Coor_hat.C[:, 1:].float()
        xyz_ref = ref_Coor_hat_coord.float()
        
        _, idx = self.knn(xyz_ref.unsqueeze(0), xyz_pred.unsqueeze(0))
        idx = idx.squeeze(0)

        q = Coor_hat.F.float()
        q = q.unsqueeze(1)
        k_raw = ref_Coor_hat_feat.float()
        k = k_raw[idx]
        

        v_raw = ref_latent_feat.float()
        v_raw = self.W_v(v_raw.transpose(0, 1).unsqueeze(0))
        v_raw = v_raw.squeeze(0).transpose(0, 1)
        v = v_raw[idx]

        attn_score = q @ k.transpose(1, 2)
        attn_score = attn_score / math.sqrt(Coor_hat.shape[-1])
        attn_score = torch.nn.functional.softmax(attn_score, dim=-1)

        pred = attn_score @ v
        pred = pred.squeeze(1)
        pred = self.W_o(pred.transpose(0, 1).unsqueeze(0))
        pred = pred.squeeze(0).transpose(0, 1)

        Feat_pred = ME.SparseTensor(
            features=pred,
            coordinate_map_key=Coor_hat.coordinate_map_key,
            coordinate_manager=Coor_hat.coordinate_manager,
            device=Coor_hat.device
        )
        Feat_pred = self.W_out(Feat_pred)
        
        return Feat_pred


class GSM(nn.Module):
    def __init__(self, in_chann=3, out_chann=16):
        super(GSM, self).__init__()
        self.module = nn.Sequential(
            ME.MinkowskiConvolution(in_chann, out_chann, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiInstanceNorm(out_chann),
            ME.MinkowskiPReLU(),
            ME.MinkowskiConvolution(out_chann, out_chann, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiInstanceNorm(out_chann),
            ME.MinkowskiPReLU(),
            ME.MinkowskiConvolution(out_chann, out_chann, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiInstanceNorm(out_chann),
            ME.MinkowskiPReLU(),
            ME.MinkowskiConvolution(out_chann, out_chann, kernel_size=1, stride=1, dimension=3),
        )
    
    def forward(self, x):
        return self.module(x)


class RTM(nn.Module):
    def __init__(self, chann, chann_mul, chann_div, type):
        super(RTM, self).__init__()
        if type == 'enc':
            self.module = nn.Sequential(
                DNConv(chann, chann, kernel_size=1, stride=1, dimension=3),
                ME.MinkowskiPReLU(),
                DNConv(chann, chann * chann_mul // chann_div, kernel_size=3, stride=1, dimension=3),
                # DNConv(chann, chann // 2, kernel_size=3, stride=1, dimension=3),
            )
        elif type == 'dec':
            self.module = nn.Sequential(
                DNConv(chann * chann_mul // chann_div, chann, kernel_size=1, stride=1, dimension=3),
                # DNConv(chann // 2, chann, kernel_size=1, stride=1, dimension=3),
                ME.MinkowskiPReLU(),
                DNConv(chann, chann , kernel_size=3, stride=1, dimension=3),
            )

    def forward(self, x):
        return self.module(x)


class DensityModule(nn.Module):
    def __init__(self, chann):
        super(DensityModule, self).__init__()
        self.module = nn.Sequential(
            ME.MinkowskiConvolution(1, chann, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiPReLU(),
            ME.MinkowskiConvolution(chann, chann, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiPReLU(),
            ME.MinkowskiConvolution(chann, chann, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiPReLU(),
            ME.MinkowskiConvolution(chann, chann, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiSigmoid()
        )
    
    def forward(self, x):
        return self.module(x)