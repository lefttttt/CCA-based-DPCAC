import math
import os
import numpy as np
import time
import MinkowskiEngine as ME

import torch
import torch.nn as nn
import torch.utils.data

from datasets.dataset import Static_Dataset
from models.network.network_I import Static_PCAC


def rgb2yuv_bt709(rgb: torch.Tensor):
    assert rgb.dtype == torch.float32
    assert rgb.ndim == 2 and rgb.shape[1] == 3
    y = (0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2])
    u = (-0.1146 * rgb[:, 0] - 0.3854 * rgb[:, 1] + 0.5000 * rgb[:, 2]) + 0.5
    v = (0.5000 * rgb[:, 0] - 0.4542 * rgb[:, 1] - 0.0458 * rgb[:, 2]) + 0.5
    return torch.cat([y[:, None], u[:, None], v[:, None]], dim=1)

def Codec(checkpoint_name, results_path, test_dataloader, channels, device):
    net_I = Static_PCAC(chann=channels).to(device)
    total_params = sum(p.numel() for p in net_I.parameters())
    print(f"Number of model parameters: {total_params:,}") 
    net_dict = torch.load(checkpoint_name, map_location=device)
    net_I.load_state_dict(net_dict, strict=True)
    net_I = net_I.to(device)

    net_I.eval()
    net_I.entropy_module.update()
    crit = nn.MSELoss()
    psnr_y_all = 0
    psnr_yuv_all = 0
    bpp_all = 0
    enc_time_all = 0
    dec_time_all = 0
    
    with torch.no_grad():
        for Batch, Data in enumerate(test_dataloader):
            xyz, color, min_xyz = Data
            bit_sum = 0
            enc_time = 0
            dec_time = 0
            pc_path = test_dataloader.dataset.data_path_list[Batch]
            pc_name = os.path.basename(pc_path)
            bin_path = os.path.join(results_path, pc_name.replace(".ply", ".bin"))

            with open(bin_path, 'wb') as fout:
                coords, feats = ME.utils.sparse_collate([xyz.squeeze(0) ], [color.squeeze(0)])
                occups = coords[:, 1:].float()/1024.0
                Input = ME.SparseTensor(
                    features=feats.float(),
                    coordinates=coords,
                    device=device,
                )
                Coor = ME.SparseTensor(
                    features=occups.float()/1024.0,
                    coordinate_map_key=Input.coordinate_map_key,
                    coordinate_manager=Input.coordinate_manager,
                    device=device,
                )
                Density = ME.SparseTensor(
                    features=torch.ones((coords.shape[0], 1)),
                    coordinate_map_key=Input.coordinate_map_key,
                    coordinate_manager=Input.coordinate_manager,
                    device=device,
                )

                start = time.time()
                latent_strings1, latent_strings2, hyper_strings = net_I.compress(Input, Coor, Density)
                enc_time = time.time() - start
                enc_time_all += enc_time

                print("Data: ", pc_name)
                print(" {} %".format(100*(len(latent_strings1[0])-len(latent_strings2[0]))/(2*len(latent_strings1[0]))))

                start = time.time()
                Output = net_I.decompress(latent_strings1, latent_strings2, hyper_strings, Coor, Density)
                dec_time = time.time() - start
                dec_time_all += dec_time

                input_colors = Input.F
                output_colors = Output.F
                output_xyz = Input.C[:, 1:] + min_xyz.to(device)

                bit_sum = (len(latent_strings1[0]) + len(latent_strings2[0]) + len(hyper_strings[0])) * 8
                num = Input.shape[0]

                fout.write(latent_strings1[0])
                fout.write(latent_strings2[0])
                fout.write(hyper_strings[0])

                org_yuv = rgb2yuv_bt709(input_colors)
                rec_yuv = rgb2yuv_bt709(output_colors)

                org_y = (org_yuv[:, 0] * 255).round().clamp(0, 255)
                rec_y = (rec_yuv[:, 0] * 255).round().clamp(0, 255)
                org_u = (org_yuv[:, 1] * 255).round().clamp(0, 255)
                rec_u = (rec_yuv[:, 1] * 255).round().clamp(0, 255)
                org_v = (org_yuv[:, 2] * 255).round().clamp(0, 255)
                rec_v = (rec_yuv[:, 2] * 255).round().clamp(0, 255)

                MSE_Y = crit(org_y, rec_y)
                MSE_U = crit(org_u, rec_u)
                MSE_V = crit(org_v, rec_v)

                loss_mse = crit(org_yuv * 255, rec_yuv * 255)

                psnr_y = 10 * torch.log10(255 * 255 / MSE_Y).item()
                psnr_u = 10 * torch.log10(255 * 255 / MSE_U).item()
                psnr_v = 10 * torch.log10(255 * 255 / MSE_V).item()

                psnr = (6 * psnr_y + psnr_u + psnr_v) / 8

                psnr_y_all += psnr_y
                psnr_yuv_all+=psnr
                bpp_all += bit_sum / num

                print(
                    "MSE all :%f  MSE Y :%f  MSE U:%f  MSE V:%f   psnr: %f   psnr_y: %f   psnr_u: %f   psnr_v: %f   bpp : %f    enc_time : %f   dec_time : %f  \n" % (
                        math.sqrt(loss_mse.item()), math.sqrt(MSE_Y.item()), math.sqrt(MSE_U.item()), math.sqrt(MSE_V.item()), psnr,
                         psnr_y, psnr_u, psnr_v, bit_sum / num, enc_time_all, dec_time_all))
                xyz = np.array(output_xyz.cpu())
                output_colors = (output_colors * 255).round().clamp(0, 255)
                feats = np.array(output_colors.cpu())
                feats = feats.astype(np.uint8)
                print(feats.shape[0])
                data = np.hstack([xyz, feats])
                ply_header = '''ply\nformat ascii 1.0\nelement vertex %(points_num)d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'''
                rec_path = os.path.join(results_path, pc_name.replace(".ply", "_rec.ply"))
                np.savetxt(rec_path, data, fmt='%f %f %f %d %d %d')
                with open(rec_path, 'r+') as f:
                    old = f.read()
                    f.seek(0)
                    f.write(ply_header % dict(points_num=len(xyz)))
                    f.write(old)
        print("---------------PSNR_Y:",psnr_y_all / len(test_dataloader), "PSNR_YUV:  ",psnr_yuv_all/ len(test_dataloader),"  Bpp:  ", bpp_all / len(test_dataloader),
               "  Enc_time: ", enc_time_all / len(test_dataloader), "  Dec_time: ", dec_time_all / len(test_dataloader))
         

if __name__ == '__main__':
    device_id = 0
    if torch.cuda.device_count() > device_id:
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    # If data-related problems occur, please check whether the path in "/datasets/dataset.py" corresponds to the point cloud sequence directory structure.
    data_path = ["/your/test/dataset/path"]
    results_path = "/your/results/path"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    test_dataset = Static_Dataset(dir_paths=data_path, mode='test', num_points=300000)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Data lenght = %d" % len(test_dataset))
    channels = 96
    checkpoint_name = "/your/static/checkpoints/name.pt"
    print("Checkpoint name = %s" % checkpoint_name)
    Codec(checkpoint_name, results_path, test_dataloader, channels, device)


