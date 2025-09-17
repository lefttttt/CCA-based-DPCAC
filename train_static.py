import time
import os
import numpy as np
import random
import datetime
import MinkowskiEngine as ME

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from models.network.network_I import Static_PCAC
from datasets.dataset import Static_Dataset


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


def train(net_I, checkpoint_all_name, train_dataloader, lam, device):

    optimizer, aux_optimizer = configure_optimizers(net_I,1e-4,1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.8)
    crit = nn.MSELoss()

    checkpoint_path = os.path.dirname(checkpoint_all_name)
    checkpoint_name = os.path.basename(checkpoint_all_name)
    if not os.path.exists(checkpoint_path):
        print("checkpoint path not exist")
        os.mkdir(checkpoint_path)
        print("made checkpoint path not exist in " + os.getcwd())

    if not os.path.exists("log"):
        print("log paprth not exist")
        os.mkdir("log")
        print("made log path not exist in " + os.getcwd())
    log_file = (os.path.join("log", checkpoint_name)).replace(".pt", ".txt")
    starttime = time.strftime("%Y-%m-%d_%H:%M:%S")

    with open(log_file, "w") as f:
        f.write(f"===== Training Started at {starttime} =====\n")
        f.write(f"Checkpoint name: {checkpoint_name}\n")
        f.write(f"lambda : {lam}\n")
        writer = SummaryWriter(log_dir="log_tensorboard/network_" + starttime, comment=starttime)
        # print(net_I)
        for epoch in range(0,150):

            net_I.train()
            MSE_sum_train = 0
            PSNR_Y_sum = 0
            bpp_latent_sum = 0
            bpp_hyper_sum = 0

            for Batch_num, Data in enumerate(train_dataloader):
                optimizer.zero_grad()
                aux_optimizer.zero_grad()
                coords, feats, occups = Data
                Input_attribute = []
                Output_attribute = []

                Input = ME.SparseTensor(
                    features=feats,
                    coordinates=coords,
                    device=device,
                )

                Coor = ME.SparseTensor(
                    features=occups[:,0:3],
                    coordinate_map_key=Input.coordinate_map_key,
                    coordinate_manager=Input.coordinate_manager,
                    device=device,
                )

                Density = ME.SparseTensor(
                    features=torch.ones((Input.F.shape[0], 1)),
                    coordinate_map_key=Input.coordinate_map_key,
                    coordinate_manager=Input.coordinate_manager,
                    device=device,
                )

                # Codec
                Output, latent_likelihoods, hyper_likelihoods = net_I(Input, Coor, Density)
                
                bpp_latent = -torch.sum(torch.log2(latent_likelihoods)) / coords.shape[0]
                bpp_hyper = -torch.sum(torch.log2(hyper_likelihoods)) / coords.shape[0]
                bpp_latent_sum += bpp_latent.item()
                bpp_hyper_sum += bpp_hyper.item()

                # Calculate the loss
                loss = torch.mean((Input.F - Output.F).pow(2)) * lam + bpp_latent + bpp_hyper

                Input_attribute = (Input.F*255).round().clamp(0, 255)
                Output_attribute = (Output.F*255).round().clamp(0, 255)
                MSE_train = crit(Input_attribute, Output_attribute)
                MSE_sum_train += MSE_train.item()
                Input_Y = rgb2yuv_bt709(Input.F[:, :3])[:, 0]*255
                Output_Y = rgb2yuv_bt709(Output.F[:, :3])[:, 0]*255
                PSNR_Y_train = 10 * torch.log10(255 * 255 / ((Input_Y - Output_Y).pow(2).mean()))
                PSNR_Y_sum += PSNR_Y_train.item()
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                aux_optimizer.step()
                
                # Log
                f.write(f"loss_sum: {torch.mean((Input.F - Output.F).pow(2)) * lam}    ")
                f.write(f"bpp_latent: {bpp_latent.item()}    ")
                f.write(f"bpp_hyper: {bpp_hyper.item()}\n")
                writer.add_scalar('MSE_train', MSE_train.item(), len(train_dataloader) * epoch + Batch_num)
                writer.add_scalar('PSNR_Y_train', PSNR_Y_train.item(), len(train_dataloader) * epoch + Batch_num)
                writer.add_scalar('bpp_latent_train', bpp_latent.item(), len(train_dataloader) * epoch + Batch_num)
                writer.add_scalar('bpp_hyper_train', bpp_hyper.item(), len(train_dataloader) * epoch + Batch_num)

                torch.cuda.empty_cache()

            print("epoch: %d  train_MSE: %f  train_PSNR_Y: %f  bpp_latent: %f  bpp_hyper: %f  lambda: %d" %
                (epoch, MSE_sum_train / len(train_dataloader), PSNR_Y_sum / len(train_dataloader),
                bpp_latent_sum / len(train_dataloader), bpp_hyper_sum / len(train_dataloader), lam))
            f.write("epoch: %d  train_MSE: %f  train_PSNR_Y: %f  bpp_latent: %f  bpp_hyper: %f  lambda: %d" %
                (epoch, MSE_sum_train / len(train_dataloader), PSNR_Y_sum / len(train_dataloader),
                bpp_latent_sum / len(train_dataloader), bpp_hyper_sum / len(train_dataloader), lam))
            f.flush()
            
            # Save the model
            torch.save(net_I.state_dict(), checkpoint_all_name)
            if (epoch+1) % 10 == 0:
                save_path = os.path.join(checkpoint_path, f"network_{epoch+1}.pt")
                torch.save(net_I.state_dict(), save_path)

            scheduler.step()


if __name__ == '__main__':
    # Initialization
    setup_seed(14)
    device_id = 0
    if torch.cuda.device_count() > device_id:
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    start_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"loading data started at: {start_time_str}")
    batch_size = 3
    lam = 650
    channels = 96
    checkpoint_name = "/your/static/checkpoints/name.pt"
    # If data-related problems occur, please check whether the path in "/datasets/dataset.py" corresponds to the point cloud sequence directory structure.
    data_path = ["/your/test/dataset/path"]
    train_dataset = Static_Dataset(dir_paths=data_path, mode='train', num_points=300000)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=ME.utils.SparseCollation(),
                                                   batch_size=batch_size, shuffle=True, num_workers=1)
    print("data lenght = %d" % len(train_dataloader))
    # Initialize the model
    net_I = Static_PCAC(chann=channels).to(device)
    total_params = sum(p.numel() for p in net_I.parameters())
    print(f"Number of model parameters: {total_params:,}")
    start_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Training started at: {start_time_str}")

    train(net_I, checkpoint_name, train_dataloader, lam, device)



