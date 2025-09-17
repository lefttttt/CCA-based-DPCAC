import torch.utils.data as data
from tqdm import tqdm
import os
from datasets.utils import *
import  open3d as o3d
import re
import numpy as np
from typing import List, Tuple, Iterator, Dict

class Static_Dataset(data.Dataset):
    def __init__(self, dir_paths, mode, num_points):
        if isinstance(dir_paths, str):
            self.dir_paths = [dir_paths]
        elif isinstance(dir_paths, list):
            self.dir_paths = dir_paths
        else:
            raise TypeError("dir_paths must be a string or a list of strings.")

        self.num_points = num_points
        self.mode = mode
        print(f"Mode: {self.mode}, num_points: {self.num_points}")

        self.data_path_list = []
        for path in self.dir_paths:
            if not os.path.isdir(path):
                print(f"Warning: Provided path '{path}' is not a valid directory. Skipping.")
                continue
            
            ply_files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".ply")]
            self.data_path_list.extend(ply_files)
        
        if not self.data_path_list:
            print("Warning: No .ply files were found in the provided directories.")
        else:
            print(f"Found a total of {len(self.data_path_list)} .ply files across all directories.")

        self.preloaded_data = None
        if self.mode == 'train':
            print("Train mode detected. Preloading all data into memory...")
            self.preloaded_data = self._preload_data()
            print("Data preloading complete.")

    def _load_and_preprocess_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        pcd = o3d.io.read_point_cloud(file_path)
        xyz = np.asarray(pcd.points).astype(np.int32)
        feats = np.asarray(pcd.colors).astype(np.float32)
        _, indices = np.unique(xyz[:, :3], axis=0, return_index=True)
        xyz = xyz[indices]
        feats = feats[indices]
        return xyz, feats

    def _preload_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        data_list = []
        for file_path in tqdm(self.data_path_list, desc="Preloading training data"):
            data_list.append(self._load_and_preprocess_file(file_path))
        return data_list

    def __getitem__(self, index):
        if self.mode == 'train' and self.preloaded_data is not None:
            xyz, feats = self.preloaded_data[index]
        else:
            file_path = self.data_path_list[index]
            xyz, feats = self._load_and_preprocess_file(file_path)
        
        xyz, (feats,) = kd_tree_partition_randomly(
            xyz,
            self.num_points,
            (feats,)
        )
        xyz_min = xyz.min(0)
        xyz -= xyz_min
        idx = morton_sort(xyz)
        xyz = xyz[idx]
        feats = feats[idx]

        if self.mode=='train':
            occups = xyz.copy().astype(np.float32)/1024.0
            return xyz, feats, occups
        else:
            return xyz, feats, xyz_min

    def __len__(self):
        return len(self.data_path_list)


class Train_Dynamic_Dataset(data.Dataset):
    def __init__(self, root_dir: str, num_points: int):
        self.root_dir = root_dir
        self.num_points = num_points
        self.box_margin = 8
        print("Creating file samples...")
        self.samples = self._create_samples()
        if not self.samples:
            raise ValueError("No valid samples found. Ensure sequences have at least 3 frames.")
        print(f"Training Dataset: Found {len(self.samples)} total valid (current, ref, ref_ref) frame groups.")

        print("Preloading all data into memory. This may take a while...")
        self.preloaded_data = self._preload_data()
        print("Data preloading complete.")

    def _create_samples(self) -> List[Tuple[str, str, str]]:
        samples = []
        seq_dirs = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        for seq_dir in seq_dirs:
            # seq_path = os.path.join(self.root_dir, seq_dir, 'Ply')
            seq_path = os.path.join(self.root_dir, seq_dir)
            if not os.path.isdir(seq_path):
                continue
                
            ply_files = sorted(
                [f for f in os.listdir(seq_path) if f.endswith(".ply")],
                key=lambda f: int(re.search(r'(\d+)(?=\.ply$)', f).group())
            )

            if len(ply_files) > 150:
                ply_files = ply_files[:150]
            
            if len(ply_files) < 3:
                continue
            
            for i in range(2, len(ply_files)):
                current_frame_path = os.path.join(seq_path, ply_files[i])
                ref_frame_path = os.path.join(seq_path, ply_files[i-1])
                ref_ref_frame_path = os.path.join(seq_path, ply_files[i-2])
                samples.append((current_frame_path, ref_frame_path, ref_ref_frame_path))
        return samples

    def load_and_preprocess(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
            pcd = o3d.io.read_point_cloud(path)
            xyz = np.asarray(pcd.points, dtype=np.int32)
            feats = np.asarray(pcd.colors, dtype=np.float32)
            _, unique_indices = np.unique(xyz, axis=0, return_index=True)
            xyz = xyz[unique_indices]
            feats = feats[unique_indices]
            return xyz, feats

    def _preload_data(self) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Loads all point cloud data from disk into a list in memory.
        """
        preloaded_list = []

        for current_path, ref_path, ref_ref_path in tqdm(self.samples, desc="Preloading Data"):
            xyz_cur, feats_cur = self.load_and_preprocess(current_path)
            xyz_ref, feats_ref = self.load_and_preprocess(ref_path)
            xyz_ref_ref, feats_ref_ref = self.load_and_preprocess(ref_ref_path)
            
            preloaded_list.append({
                'cur': (xyz_cur, feats_cur),
                'ref': (xyz_ref, feats_ref),
                'ref_ref': (xyz_ref_ref, feats_ref_ref),
            })
        return preloaded_list

    def __getitem__(self, index: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        preloaded_sample = self.preloaded_data[index]

        xyz_cur, feats_cur = preloaded_sample['cur']
        xyz_ref_full, feats_ref_full = preloaded_sample['ref']
        xyz_ref_ref_full, feats_ref_ref_full = preloaded_sample['ref_ref']
        
        xyz_cur, (feats_cur,) = kd_tree_partition_randomly(
            xyz_cur, self.num_points, (feats_cur,)
        )

        box_min = xyz_cur.min(0) - self.box_margin
        box_max = xyz_cur.max(0) + self.box_margin

        mask_ref = np.all((xyz_ref_full >= box_min) & (xyz_ref_full <= box_max), axis=1)
        xyz_ref_crop, feats_ref_crop = xyz_ref_full[mask_ref], feats_ref_full[mask_ref]
        
        mask_ref_ref = np.all((xyz_ref_ref_full >= box_min) & (xyz_ref_ref_full <= box_max), axis=1)
        xyz_ref_ref_crop, feats_ref_ref_crop = xyz_ref_ref_full[mask_ref_ref], feats_ref_ref_full[mask_ref_ref]

        xyz_cur = xyz_cur - box_min
        idx_cur = morton_sort(xyz_cur)
        xyz_cur, feats_cur = xyz_cur[idx_cur], feats_cur[idx_cur]
        
        if xyz_ref_crop.shape[0] > 0:
            xyz_ref = xyz_ref_crop - box_min
            feats_ref = feats_ref_crop
            idx_ref = morton_sort(xyz_ref)
            xyz_ref, feats_ref = xyz_ref[idx_ref], feats_ref[idx_ref]
        else: 
            xyz_ref = np.empty((0, 3), dtype=np.int32)
            feats_ref = np.empty((0, 3), dtype=np.float32)

        if xyz_ref_ref_crop.shape[0] > 0:
            xyz_ref_ref = xyz_ref_ref_crop - box_min
            feats_ref_ref = feats_ref_ref_crop
            idx_ref_ref = morton_sort(xyz_ref_ref)
            xyz_ref_ref, feats_ref_ref = xyz_ref_ref[idx_ref_ref], feats_ref_ref[idx_ref_ref]
        else:
            xyz_ref_ref = np.empty((0, 3), dtype=np.int32)
            feats_ref_ref = np.empty((0, 3), dtype=np.float32)
        
        return {
            'cur': (xyz_cur, feats_cur),
            'ref': (xyz_ref, feats_ref),
            'ref_ref': (xyz_ref_ref, feats_ref_ref),
        }

    def __len__(self):
        return len(self.samples)


class Test_Dynamic_Dataset(data.Dataset):
    """
    A dataset for testing dynamic point cloud sequences.
    It handles reading, cleaning, and partitioning the point clouds.
    """
    def __init__(self, root_dir: str, sub_pc_point_num: int = 500000):
        super().__init__()
        self.root_dir = root_dir
        self.max_num_points = sub_pc_point_num
        self.box_margin = 8
        
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Test data directory not found at: {root_dir}")
        
        self.sequences = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        
        if not self.sequences:
            raise ValueError(f"No sequences found in {root_dir}")
            
        print(f"Found {len(self.sequences)} test sequences. Partitions will have max {self.max_num_points} points.")
    
    @staticmethod
    def _recursive_kd_partition(xyz: np.ndarray, feats: np.ndarray, max_points: int) -> List[Dict[str, np.ndarray]]:
        """
        The core recursive function for KD-Tree partitioning.
        """
        if len(xyz) <= max_points:
            # Morton sorting of sub-point clouds
            idx = morton_sort(xyz)
            xyz,  feats = xyz[idx], feats[idx]
            return [{'coords': xyz, 'feats': feats}]

        # 1. Find the axis with the largest spread (variance)
        split_axis = np.argmax(np.ptp(xyz, axis=0)) # ptp = peak-to-peak (max - min)
        
        # 2. Find the median along this axis
        median = np.median(xyz[:, split_axis])
        
        # 3. Split the points
        mask = xyz[:, split_axis] <= median
        
        # Handle cases where the median split is not effective (e.g., many identical points)
        if not np.any(mask) or not np.any(~mask):
            # Fallback to a simple split if median split fails
            half = len(xyz) // 2
            mask = np.zeros(len(xyz), dtype=bool)
            mask[:half] = True

        xyz1, feats1 = xyz[mask], feats[mask]
        xyz2, feats2 = xyz[~mask], feats[~mask]
        
        # 4. Recurse on the two halves and combine the results
        partitions1 = Test_Dynamic_Dataset._recursive_kd_partition(xyz1, feats1, max_points)
        partitions2 = Test_Dynamic_Dataset._recursive_kd_partition(xyz2, feats2, max_points)
        
        return partitions1 + partitions2

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[str, Iterator[Tuple[List[Dict[str, np.ndarray]], np.ndarray, str]]]:
        """
        Returns an iterator for the frames of a specific sequence.
        Each item yielded by the iterator contains a list of partitions for that frame.
        """
        seq_path = self.sequences[index]
        seq_name = os.path.basename(seq_path)
        # ply_files = sorted([os.path.join(seq_path, f) for f in os.listdir(seq_path) if f.endswith(".ply")],
        #                 key=lambda f: int(re.search(r'(\d+)(?=\.ply$)', os.path.basename(f)).group()))
        ply_files = sorted(
            [os.path.join(seq_path, f) for f in os.listdir(seq_path) if f.endswith(".ply")],
            key=lambda f: int(
                re.search(r'(\d+)(?:_recolor)?\.ply$', os.path.basename(f)).group(1)
            )
        )

        def frame_generator():
            for file_path in ply_files:
                pcd = o3d.io.read_point_cloud(file_path)
                xyz = np.asarray(pcd.points, dtype=np.int32)
                feats = np.asarray(pcd.colors, dtype=np.float32)

                _, unique_indices = np.unique(xyz, axis=0, return_index=True)
                xyz, feats = xyz[unique_indices], feats[unique_indices]
                
                min_xyz = xyz.min(0) - self.box_margin
                xyz -= min_xyz

                partitions = self._recursive_kd_partition(xyz, feats, self.max_num_points)
                
                yield partitions, xyz, feats, min_xyz, self.box_margin, file_path

        return seq_name, frame_generator()
    
