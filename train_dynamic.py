import os
import sys
import numpy as np
import random
import time
import MinkowskiEngine as ME
import datetime
import copy
from typing import List, Dict

import torch
import torch.nn as nn
import torch.utils.data

from models.network.network_P import Dynamic_PCAC
from datasets.dataset import Train_Dynamic_Dataset

def bitrate_select(bitrate_point):
    if bitrate_point == 'r1':
        channels = 96; channels_mul = 3; channels_div = 4; lam = 200
    elif bitrate_point == 'r2':
        channels = 96; channels_mul = 7; channels_div = 8; lam = 500
    elif bitrate_point == 'r3':
        channels = 96; channels_mul = 7; channels_div = 8; lam = 1000
    elif bitrate_point == 'r4':
        channels = 96; channels_mul = 1; channels_div = 1; lam = 3000
    else:
        sys.exit(f"{bitrate_point} is is an undefined bitrate point.")

    return channels, channels_mul, channels_div, lam

def configure_optimizers(net, learning_rate,aux_learning_rate):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = set(
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    )
    aux_parameters = set(
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    )

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    # assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = torch.optim.Adam(
        (params_dict[n] for n in sorted(list(parameters))),
        lr=learning_rate,
    )
    aux_optimizer = torch.optim.Adam(
        (params_dict[n] for n in sorted(list(aux_parameters))),
        lr=aux_learning_rate,
    )
    return optimizer, aux_optimizer


def custom_collate_fn(batch: List[Dict]):
    collated_batch = {'cur': {}, 'ref': {}, 'ref_ref': {}}
    
    cur_data_list = [item['cur'] for item in batch]
    ref_data_list = [item['ref'] for item in batch]
    ref_ref_data_list = [item['ref_ref'] for item in batch]

    cur_xyzs, cur_feats = zip(*cur_data_list)
    ref_xyzs, ref_feats = zip(*ref_data_list)
    ref_ref_xyzs, ref_ref_feats = zip(*ref_ref_data_list)
    
    cur_coords, cur_colors = ME.utils.sparse_collate(list(cur_xyzs), list(cur_feats))
    ref_coords, ref_colors = ME.utils.sparse_collate(list(ref_xyzs), list(ref_feats))
    ref_ref_coords, ref_ref_colors = ME.utils.sparse_collate(list(ref_ref_xyzs), list(ref_ref_feats))
    cur_occups = cur_coords[:, 1:].float()/1024.0
    ref_occups = ref_coords[:, 1:].float()/1024.0
    ref_ref_occups = ref_ref_coords[:, 1:].float()/1024.0
    
    collated_batch = {
        'cur': {
            'coords': cur_coords,
            'colors': cur_colors,
            'occups': cur_occups
        },
        'ref': {
            'coords': ref_coords,
            'colors': ref_colors,
            'occups': ref_occups
        },
        'ref_ref': {
            'coords': ref_ref_coords,
            'colors': ref_ref_colors,
            'occups': ref_ref_occups
        }
    }
    
    return collated_batch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def rgb2yuv_bt709(rgb: torch.Tensor):
    assert rgb.dtype == torch.float32
    assert rgb.ndim == 2 and rgb.shape[1] == 3
    y = (0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2])
    u = (-0.1146 * rgb[:, 0] - 0.3854 * rgb[:, 1] + 0.5000 * rgb[:, 2]) + 0.5
    v = (0.5000 * rgb[:, 0] - 0.4542 * rgb[:, 1] - 0.0458 * rgb[:, 2]) + 0.5
    return torch.cat([y[:, None], u[:, None], v[:, None]], dim=1)

@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float = 0.999):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

def ema_decay_schedule(step, burn_in_steps, d_start = 0.98, d_end = 0.999) -> float:
    r = min(1.0, step / burn_in_steps)
    return d_start + (d_end - d_start) * r

def linear_schedule(current_step, warm_steps, burn_in_steps, start = 0.0, end = 1.0):
    if current_step < burn_in_steps:
        return 0.0
    if warm_steps <= 0:
        return end
    ratio = min(1.0, current_step / float(warm_steps))
    return start + (end - start) * ratio

def train(net_P, checkpoint_all_name, train_dataloader, lam, epoch_num, device):
    # Initial teacher network
    net_T = Dynamic_PCAC(chann=net_P.chann, chann_mul=net_P.chann_mul, chann_div=net_P.chann_div)
    net_T.load_state_dict(net_P.state_dict(), strict=True)
    net_T = net_T.to(device)
    for p in net_T.parameters():
        p.requires_grad_(False)
    net_T.eval()
    global_step = 0
    steps_per_epoch = max(1, len(train_dataloader))
    warm_ref_epochs = max(1, int(0.4 * epoch_num))  # The first 40% is warmed up
    burn_in_steps = 1 * steps_per_epoch
    warm_steps = warm_ref_epochs * steps_per_epoch

    optimizer, aux_optimizer = configure_optimizers(net_P,1e-4,1e-3)
     # The optimizer can be changed to reduce the original speed by 0.8 times every 12 rounds, for a total of 120 rounds
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, (epoch_num//10), gamma=0.8)
    crit = nn.MSELoss()

    checkpoint_path = os.path.dirname(checkpoint_all_name)
    checkpoint_name = os.path.basename(checkpoint_all_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs("log", exist_ok=True)
    log_file = (os.path.join("log", checkpoint_name)).replace(".pt", ".txt")
    starttime = time.strftime("%Y-%m-%d_%H:%M:%S")

    with open(log_file, "w") as f:
        f.write(f"===== Training Started at {starttime} =====\n")
        f.write(f"Checkpoint name: {checkpoint_name}\n")
        f.write(f"lambda : {lam}\n")
        print(net_P)
        for epoch in range(epoch_num):
            net_P.train()
            MSE_sum_train = 0
            PSNR_Y_sum = 0
            bpp_latent_sum = 0
            bpp_hyper_sum = 0

            for Batch_num, Data in enumerate(train_dataloader):
                optimizer.zero_grad()
                aux_optimizer.zero_grad()
                # Data preprocessing
                cur_coords = Data['cur']['coords'].to(device)
                cur_feats = Data['cur']['colors'].to(device)
                cur_occups = Data['cur']['occups'].to(device)
                ref_coords = Data['ref']['coords'].to(device)
                ref_feats = Data['ref']['colors'].to(device)
                ref_occups = Data['ref']['occups'].to(device)
                ref_ref_coords = Data['ref_ref']['coords'].to(device)
                ref_ref_feats = Data['ref_ref']['colors'].to(device)
                ref_ref_occups = Data['ref_ref']['occups'].to(device)
                Input_attribute = []
                Output_attribute = []
                # Attribute information
                Input = ME.SparseTensor(
                    features=cur_feats.float(),
                    coordinates=cur_coords,
                    device=device,
                )
                ref_Input = ME.SparseTensor(
                    features=ref_feats.float(),
                    coordinates=ref_coords,
                    device=device,
                )
                ref_ref_Input = ME.SparseTensor(
                    features=ref_ref_feats.float(),
                    coordinates=ref_ref_coords,
                    device=device,
                )

                # Coordinate information
                Coor = ME.SparseTensor(
                    features=cur_occups.float(),
                    coordinate_map_key=Input.coordinate_map_key,
                    coordinate_manager=Input.coordinate_manager,
                    device=device,
                )
                ref_Coor = ME.SparseTensor(
                    features=ref_occups.float(),
                    coordinate_map_key=ref_Input.coordinate_map_key,
                    coordinate_manager=ref_Input.coordinate_manager,
                    device=device,
                )
                ref_ref_Coor = ME.SparseTensor(
                    features=ref_ref_occups.float(),
                    coordinate_map_key=ref_ref_Input.coordinate_map_key,
                    coordinate_manager=ref_ref_Input.coordinate_manager,
                    device=device,
                )

                # Calculate scheduling parameters
                p_use_dec_ref = linear_schedule(global_step, warm_steps, burn_in_steps, start=0.0, end=1.0)
                use_decoded_ref = (random.random() < p_use_dec_ref)
                decay = ema_decay_schedule(global_step, burn_in_steps, d_start=0.98, d_end=0.999)

                # Generate reconstruction reference by teacher network or use GT reference
                if use_decoded_ref:
                    with torch.no_grad():
                        dec_ref = net_T.generate_dec_ref(ref_Input, ref_ref_Input, ref_Coor, ref_ref_Coor)
                    external_Output_ref = dec_ref
                else:
                    external_Output_ref = ref_Input  # use GT reference
                
                # Codec
                Output, latent_likelihoods, hyper_likelihoods = net_P(
                            Input, ref_Input, ref_ref_Input, Coor, ref_Coor, ref_ref_Coor, external_Output_ref)

                # Calculate the Bpp
                bpp_latent = -torch.sum(torch.log2(latent_likelihoods)) / cur_coords.shape[0]
                bpp_hyper = -torch.sum(torch.log2(hyper_likelihoods)) / cur_coords.shape[0]
                bpp_latent_sum += bpp_latent.item()
                bpp_hyper_sum += bpp_hyper.item()

                # Calculate the loss
                loss = torch.mean((Input.F - Output.F).pow(2)) * lam + bpp_latent + bpp_hyper

                Input_attribute = (Input.F*255).round().clamp(0, 255)
                Output_attribute = (Output.F*255).round().clamp(0, 255)
                MSE_train = crit(Input_attribute, Output_attribute)
                MSE_sum_train += MSE_train.item()
                Input_Y = (rgb2yuv_bt709(Input.F[:, :3])[:, 0]*255).round().clamp(0, 255)
                Output_Y = (rgb2yuv_bt709(Output.F[:, :3])[:, 0]*255).round().clamp(0, 255)
                PSNR_Y_train = 10 * torch.log10(255 * 255 / ((Input_Y - Output_Y).pow(2).mean()))
                PSNR_Y_sum += PSNR_Y_train.item()
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                aux_optimizer.step()

                # EMA update techer network
                ema_update(net_T, net_P, decay=decay)
                
                # Log
                f.write(f"loss_sum: {(torch.mean((Input.F - Output.F).pow(2)) * lam).item()}    ")
                f.write(f"bpp_latent: {bpp_latent.item()}    ")
                f.write(f"bpp_hyper: {bpp_hyper.item()}\n")

                global_step += 1
                torch.cuda.empty_cache()

            print("epoch: %d  train_MSE: %f  train_PSNR_Y: %f  bpp_latent: %f  bpp_hyper: %f  lambda: %d" %
                (epoch, MSE_sum_train / len(train_dataloader), PSNR_Y_sum / len(train_dataloader),
                bpp_latent_sum / len(train_dataloader), bpp_hyper_sum / len(train_dataloader), lam))
            f.write("epoch: %d  train_MSE: %f  train_PSNR_Y: %f  bpp_latent: %f  bpp_hyper: %f  lambda: %d" %
                (epoch, MSE_sum_train / len(train_dataloader), PSNR_Y_sum / len(train_dataloader),
                bpp_latent_sum / len(train_dataloader), bpp_hyper_sum / len(train_dataloader), lam))
            f.flush()

            # Save the model
            torch.save(net_P.state_dict(), checkpoint_all_name)
            if (epoch + 1) % 10 == 0:
                save_path = os.path.join(checkpoint_path, f"network_{epoch+1}.pt")
                torch.save(net_T.state_dict(), save_path)

            scheduler.step()


if __name__ == '__main__':
    # Initialization
    setup_seed(14)
    device_id = 0
    bitrate_point = 'r1' # r1~r4
    print(f"Start training the {bitrate_point} model")
    channels, channels_mul, channels_div, lam = bitrate_select(bitrate_point=bitrate_point)
    if torch.cuda.device_count() > device_id:
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    start_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"loading data started at: {start_time_str}")
    batch_size = 3
    epoch_num = 200
    checkpoint_name = f"/your/dynamic/checkpoints/path/checkpoints_d_{bitrate_point}/network_dynamic_{bitrate_point}.pt"
    # If data-related problems occur, please check whether the path in "/datasets/dataset.py" corresponds to the point cloud sequence directory structure.
    data_path = "/your/train/dataset/path"
    # Initialize the dataset
    train_dataset = Train_Dynamic_Dataset(root_dir=data_path, num_points=300000)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=custom_collate_fn,
                                                   batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    # Initialize the model
    net_P = Dynamic_PCAC(chann=channels, chann_mul=channels_mul, chann_div=channels_div).to(device)
    total_params = sum(p.numel() for p in net_P.parameters())
    print(f"Number of model parameters: {total_params:,}") 
    start_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Training started at: {start_time_str}")

    train(net_P, checkpoint_name, train_dataloader, lam, epoch_num, device)

    end_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Training finished at: {end_time_str}")