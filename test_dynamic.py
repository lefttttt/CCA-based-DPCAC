import os
import sys
import time
import numpy as np
import struct
import open3d as o3d
from pc_error import compute_metrics
from scipy.spatial import cKDTree
import MinkowskiEngine as ME
from typing import  Dict, List, Tuple

import torch
import torch.nn as nn

from models.network.network_P import Dynamic_PCAC
from models.network.network_I import Static_PCAC
from datasets.dataset import Test_Dynamic_Dataset

def Model_election(checkpoint_dynamic_name):
    if 'r1' in checkpoint_dynamic_name:
        channels = 96; channels_mul = 3; channels_div = 4
    elif 'r2' in checkpoint_dynamic_name:
        channels = 96; channels_mul = 7; channels_div = 8
    elif 'r3' in checkpoint_dynamic_name:
        channels = 96; channels_mul = 7; channels_div = 8
    elif 'r4' in checkpoint_dynamic_name:
        channels = 96; channels_mul = 1; channels_div = 1
    else:
        sys.exit(f"{checkpoint_dynamic_name} does not match the existing bitrate point.")
    
    return channels, channels_mul, channels_div

def gen_local_cache(ref_cache: Dict[str, torch.Tensor], min_xyz_tensor: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    local_cache = {}
    ref_coords = ref_cache['coords'].to(device)
    mask = (ref_coords >= box_min) & (ref_coords <= box_max)
    mask = mask.all(dim=1)
    if not mask.any():
        return None
    for key, tensor in ref_cache.items():
        if tensor.shape[0] == ref_coords.shape[0]:
            local_cache[key] = tensor[mask].to(device)
        else:
            local_cache[key] = tensor.to(device)

    local_coords = local_cache['coords'] - min_xyz_tensor
    ref_pc_num = local_coords.shape[0]
    batch_indices = torch.zeros(ref_pc_num, 1, dtype=torch.int32).to(device)
    local_cache['coords'] = torch.cat([batch_indices, local_coords], dim=1).int()

    return local_cache

def rgb2yuv_bt709(rgb: torch.Tensor):
    assert rgb.dtype == torch.float32
    assert rgb.ndim == 2 and rgb.shape[1] == 3
    y = (0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2])
    u = (-0.1146 * rgb[:, 0] - 0.3854 * rgb[:, 1] + 0.5000 * rgb[:, 2]) + 0.5
    v = (0.5000 * rgb[:, 0] - 0.4542 * rgb[:, 1] - 0.0458 * rgb[:, 2]) + 0.5
    return torch.cat([y[:, None], u[:, None], v[:, None]], dim=1)

def write_ply(path, xyz, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd, write_ascii=True)

def write_bitstream(file_handle, num_partitions: int, bitstream_per_partition: List[Tuple[bytes, bytes, bytes]]):
    file_handle.write(struct.pack('I', num_partitions))
    all_lengths = []
    for part_streams in bitstream_per_partition:
        all_lengths.extend([len(s) for s in part_streams])
    for length in all_lengths:
        file_handle.write(struct.pack('I', length))
    for part_streams in bitstream_per_partition:
        for stream in part_streams:
            file_handle.write(stream)


def Codec(checkpoint_dynamic_path, checkpoint_static_path, results_root_path, test_dataset, Code_mode, GOF, max_frame_idx, device):
    net_I = Static_PCAC(chann=96).to(device)
    total_static_params = sum(p.numel() for p in net_I.parameters())
    print(f"Number of static model parameters: {total_static_params:,}")
    net_I_dict = torch.load(checkpoint_static_path, map_location=device)
    net_I.load_state_dict(net_I_dict, strict=True)
    print(f"Loading static checkpoint from: {checkpoint_static_path}")
    
    channels, channels_mul, channels_div = Model_election(os.path.basename(checkpoint_dynamic_path))
    net_P = Dynamic_PCAC(chann=channels, chann_mul=channels_mul, chann_div=channels_div).to(device)
    total_dynamic_params = sum(p.numel() for p in net_P.parameters())
    print(f"Number of model parameters: {total_dynamic_params:,}") 
    net_P_dict = torch.load(checkpoint_dynamic_path, map_location=device)
    net_P.load_state_dict(net_P_dict, strict=True)
    print(f"Loading dynamic checkpoint from: {checkpoint_dynamic_path}")

    net_I.eval()
    net_I.entropy_module.update()
    net_P.eval()
    net_P.entropy_module.update()
    crit = nn.MSELoss()
    results_file = os.path.join(results_root_path, "results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Results for Dynamic PCAC\n")
        f.write(f"Dynamic Checkpoint: {checkpoint_dynamic_path}\n")
        f.write(f"Static Checkpoint: {checkpoint_static_path}\n")
        f.write(f"Channels: {channels}\n")
        f.write(f"Device: {device}\n\n")

        with torch.no_grad():
            for seq_idx in range(len(test_dataset)):
                seq_psnr_y, seq_psnr_u, seq_psnr_v, seq_psnr_yuv, seq_bpp = 0, 0, 0, 0, 0
                seq_enc_time, seq_dec_time, seq_frames = 0, 0, 0
                seq_name, frame_iterator = test_dataset[seq_idx]
                seq_results_path = os.path.join(results_root_path, seq_name)
                os.makedirs(seq_results_path, exist_ok=True)
                print(f"\n--- Processing Sequence: {seq_name} ---")
                f.write(f"Processing Sequence: {seq_name}\n")
                ref_cache = None
                for frame_idx, (partitions, orig_xyz, orig_color, min_xyz, box_margin, frame_path) in enumerate(frame_iterator):
                    # For test, we only process the first 16 frames
                    frame_name = os.path.basename(frame_path)
                    if frame_idx >= max_frame_idx:
                        break
                    if ('viewdep' in seq_name) and (frame_idx >= 60):
                        break
                    min_xyz_tensor = torch.from_numpy(min_xyz).float().to(device)
                    frame_enc_time, frame_dec_time, frame_total_bits = 0, 0, 0
                    all_recon_coords, all_recon_feats, frame_bitstream_parts = [], [], []

                    for part_idx, part_data in enumerate(partitions):
                        # Data pre-processing
                        part_coords, part_feats = part_data['coords'], part_data['feats']
                        coords, feats = ME.utils.sparse_collate([part_coords], [part_feats])
                        occups  = coords[:, 1:].float()/1024.0
                        Input = ME.SparseTensor(
                            features=feats.float(),
                            coordinates=coords.int(),
                            device=device
                        )
                        Coor = ME.SparseTensor(
                            features=occups,
                            coordinate_map_key=Input.coordinate_map_key,
                            coordinate_manager=Input.coordinate_manager,
                            device=device
                        )
                        

                        # I-frame processing
                        if Code_mode==0 or (Code_mode==1 and frame_idx==0) or (Code_mode==2 and frame_idx%GOF==0):
                            # Encode
                            Density = ME.SparseTensor(
                            features=torch.ones(coords.shape[0], 1),
                            coordinate_map_key=Input.coordinate_map_key,
                            coordinate_manager=Input.coordinate_manager,
                            device=device
                            )
                            start_enc = time.time()
                            latent_strings1, latent_strings2, hyper_strings = net_I.compress(Input, Coor, Density)
                            frame_enc_time = time.time() - start_enc
                            # print(f"Data: {frame_name} | Entropy encode improvement: {100*(len(latent_strings1[0])-len(latent_strings2[0]))/(2*len(latent_strings1[0])):.4f}%")
                            
                            # Decode
                            start_dec = time.time()
                            Output = net_I.decompress(latent_strings1, latent_strings2, hyper_strings, Coor, Density)
                            frame_dec_time = time.time() - start_dec
                        # P-frame processing
                        else:
                            # (.min()-box_margin, .max()+box_margin) is used to ensure the bounding box is large enough to include all points
                            box_min = Input.C[:, 1:].min(dim=0).values + min_xyz_tensor - box_margin
                            box_max = Input.C[:, 1:].max(dim=0).values + min_xyz_tensor + box_margin
                            local_ref_cache = gen_local_cache(ref_cache, min_xyz_tensor, box_min, box_max, device)
                            # Encode
                            start_enc = time.time()
                            latent_strings1, latent_strings2, hyper_strings = net_P.compress(Input, Coor, local_ref_cache)
                            frame_enc_time = time.time() - start_enc
                            # print(f"Data: {frame_name} | Entropy encode improvement: {100*(len(latent_strings1[0])-len(latent_strings2[0]))/(2*len(latent_strings1[0])):.4f}%")

                            # Decode
                            start_dec = time.time()
                            Output = net_P.decompress(latent_strings1, latent_strings2, hyper_strings, Coor, local_ref_cache)
                            frame_dec_time = time.time() - start_dec
                            frame_enc_time += frame_dec_time

                        # Results processing
                        part_streams = (latent_strings1[0], latent_strings2[0], hyper_strings[0])
                        frame_bitstream_parts.append(part_streams)
                        frame_total_bits += sum(len(group) for group in part_streams) * 8
                        all_recon_coords.append(Output.C[:, 1:])
                        all_recon_feats.append(Output.F)
                    
                    full_recon_coords_local = torch.cat(all_recon_coords, dim=0)
                    full_recon_coords = full_recon_coords_local + min_xyz_tensor
                    full_recon_feats = torch.cat(all_recon_feats, dim=0)
                    ref_cache = {
                        'coords': full_recon_coords,
                        'feats': full_recon_feats
                    }

                    bpp = frame_total_bits / len(orig_xyz)
                    seq_bpp += bpp
                    seq_enc_time += frame_enc_time
                    seq_dec_time += frame_dec_time
                    seq_frames += 1

                    # PSNR is calculated on the full point cloud
                    org_yuv = rgb2yuv_bt709(torch.from_numpy(orig_color).float()).to(device)
                    
                    # Match the original point cloud with the reconstructed point cloud
                    tree = cKDTree(full_recon_coords_local.cpu().numpy())
                    _, matched_indices = tree.query(orig_xyz, k=1)
                    rec_yuv = rgb2yuv_bt709(full_recon_feats[matched_indices])

                    mse_y = crit(org_yuv[:, 0], rec_yuv[:, 0])
                    mse_u = crit(org_yuv[:, 1], rec_yuv[:, 1])
                    mse_v = crit(org_yuv[:, 2], rec_yuv[:, 2])

                    psnr_y = 10 * torch.log10(1 / mse_y).item()
                    psnr_u = 10 * torch.log10(1 / mse_u).item()
                    psnr_v = 10 * torch.log10(1 / mse_v).item()
                    psnr_yuv = (6 * psnr_y + psnr_u + psnr_v) / 8
                    
                    seq_psnr_y += psnr_y
                    seq_psnr_u += psnr_u
                    seq_psnr_v += psnr_v
                    seq_psnr_yuv += psnr_yuv
                    
                    print(f"Frame: {frame_name} | PSNR-Y: {psnr_y:.4f} | PSNR-U: {psnr_u:.4f} | PSNR-V: {psnr_v:.4f} | PSNR-YUV: {psnr_yuv:.4f} | BPP: {bpp:.4f} | EncT: {frame_enc_time:.4f}s | DecT: {frame_dec_time:.4f}s")
                    f.write(f"Frame: {frame_name} | PSNR-Y: {psnr_y:.4f} | PSNR-U: {psnr_u:.4f} | PSNR-V: {psnr_v:.4f} | PSNR-YUV: {psnr_yuv:.4f} | BPP: {bpp:.4f} | EncT: {frame_enc_time:.4f}s | DecT: {frame_dec_time:.4f}s\n")
                    # Save results
                    bin_path = os.path.join(seq_results_path, frame_name.replace('.ply', '.bin'))
                    with open(bin_path, 'wb') as bf:
                        write_bitstream(bf, len(partitions), frame_bitstream_parts)

                    rec_path = os.path.join(seq_results_path, frame_name.replace('.ply', '_rec.ply'))
                    write_ply(rec_path, (full_recon_coords.cpu().numpy()), full_recon_feats.cpu().numpy())

                avg_psnr_y = seq_psnr_y / seq_frames
                avg_psnr_u = seq_psnr_u / seq_frames
                avg_psnr_v = seq_psnr_v / seq_frames
                avg_psnr_yuv = seq_psnr_yuv / seq_frames
                avg_bpp = seq_bpp / seq_frames
                avg_enc_time = seq_enc_time / seq_frames
                avg_dec_time = seq_dec_time / seq_frames

                print(f"\n================== {seq_name} Average Results ==================")
                print(f"Total Frames Processed: {seq_frames}")
                print(f"Average PSNR-Y:   {avg_psnr_y:.4f} dB")
                print(f"Average PSNR-U:   {avg_psnr_u:.4f} dB")
                print(f"Average PSNR-V:   {avg_psnr_v:.4f} dB")
                print(f"Average PSNR-YUV: {avg_psnr_yuv:.4f} dB")
                print(f"Average BPP:      {avg_bpp:.4f} bpp")
                print(f"Average Enc Time: {avg_enc_time:.4f} s")
                print(f"Average Dec Time: {avg_dec_time:.4f} s")
                print("=========================================================\n")
                f.write(f"\n================== {seq_name} Average Results ==================\n")
                f.write(f"Total Frames Processed: {seq_frames}\n")
                f.write(f"Average PSNR-Y:   {avg_psnr_y:.4f} dB\n")
                f.write(f"Average PSNR-U:   {avg_psnr_u:.4f} dB\n")
                f.write(f"Average PSNR-V:   {avg_psnr_v:.4f} dB\n")
                f.write(f"Average PSNR-YUV: {avg_psnr_yuv:.4f} dB\n")
                f.write(f"Average BPP:      {avg_bpp:.4f} bpp\n")
                f.write(f"Average Enc Time: {avg_enc_time:.4f} s\n")
                f.write(f"Average Dec Time: {avg_dec_time:.4f} s\n")
                f.write("=========================================================\n")


if __name__ == '__main__':
    device_id = 0
    if torch.cuda.device_count() > device_id:
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    # If data-related problems occur, please check whether the path in "/datasets/dataset.py" corresponds to the point cloud sequence directory structure.
    data_path = "/your/test/dataset/path"
    results_path = "/your/results/path"
    os.makedirs(results_path, exist_ok=True)

    checkpoint_dynamic_name = "/your/dynamic/checkpoints/name.pt"
    checkpoint_static_name = "/your/static/checkpoints/name.pt"
    Code_mode = 1 # 0: All-Intra 1: Low-Delay 2: Random Access
    if Code_mode==2:
        GOF = 30 # Group of frames
    else:
        GOF = None
    max_frame_idx = 60 # Limit the number of frames to test in a sequence
    test_dataset = Test_Dynamic_Dataset(root_dir=data_path, sub_pc_point_num=300000)
    
    Codec(checkpoint_dynamic_name, checkpoint_static_name, results_path, test_dataset, Code_mode, GOF, max_frame_idx, device)