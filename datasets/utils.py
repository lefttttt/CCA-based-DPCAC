
from typing import Tuple, List, Optional, Union, Callable
import numpy as np
import torch

def morton_sort(xyz):
    vx,vy,vz = xyz[:,2],xyz[:,1],xyz[:,0]
    vx = vx.astype('int64')
    vy = vy.astype('int64')
    vz = vz.astype('int64')
    val = ((0x000001 & vx)) + ((0x000001 & vy) << 1) + ((0x000001 & vz) << 2) + \
          ((0x000002 & vx) << 2) + ((0x000002 & vy) << 3) + ((0x000002 & vz) << 4) + \
          ((0x000004 & vx) << 4) + ((0x000004 & vy) << 5) + ((0x000004 & vz) << 6) + \
          ((0x000008 & vx) << 6) + ((0x000008 & vy) << 7) + ((0x000008 & vz) << 8) + \
          ((0x000010 & vx) << 8) + ((0x000010 & vy) << 9) + ((0x000010 & vz) << 10) + \
          ((0x000020 & vx) << 10) + ((0x000020 & vy) << 11) + ((0x000020 & vz) << 12) + \
          ((0x000040 & vx) << 12) + ((0x000040 & vy) << 13) + ((0x000040 & vz) << 14) + \
          ((0x000080 & vx) << 14) + ((0x000080 & vy) << 15) + ((0x000080 & vz) << 16) + \
          ((0x000100 & vx) << 16) + ((0x000100 & vy) << 17) + ((0x000100 & vz) << 18) + \
          ((0x000200 & vx) << 18) + ((0x000200 & vy) << 19) + ((0x000200 & vz) << 20) + \
          ((0x000400 & vx) << 20) + ((0x000400 & vy) << 21) + ((0x000400 & vz) << 22) + \
          ((0x000800 & vx) << 22) + ((0x000800 & vy) << 23) + ((0x000800 & vz) << 24) + \
          ((0x001000 & vx) << 24) + ((0x001000 & vy) << 25) + ((0x001000 & vz) << 26) + \
          ((0x002000 & vx) << 26) + ((0x002000 & vy) << 27) + ((0x002000 & vz) << 28) + \
          ((0x004000 & vx) << 28) + ((0x004000 & vy) << 29) + ((0x004000 & vz) << 30) + \
          ((0x008000 & vx) << 30) + ((0x008000 & vy) << 31) + ((0x008000 & vz) << 32) + \
          ((0x010000 & vx) << 32) + ((0x010000 & vy) << 33) + ((0x010000 & vz) << 34) + \
          ((0x020000 & vx) << 34) + ((0x020000 & vy) << 35) + ((0x020000 & vz) << 36) + \
          ((0x040000 & vx) << 36) + ((0x040000 & vy) << 37) + ((0x040000 & vz) << 38) + \
          ((0x080000 & vx) << 38) + ((0x080000 & vy) << 39) + ((0x080000 & vz) << 40) + \
          ((0x100000 & vx) << 40) + ((0x100000 & vy) << 41) + ((0x100000 & vz) << 42)
    idx = np.argsort(val)
    return  idx

def kd_tree_partition_randomly(
        data: np.ndarray, target_num: int, extras: Tuple[Optional[np.ndarray], ...] = (),
        choice_fn: Callable[[np.ndarray], int] = lambda d: np.argmax(np.var(d, 0)).item(),
        cur_target_num_scaler: float = 0.5
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[Optional[np.ndarray], ...]]]:
    len_data = len(data)
    if len_data <= target_num:
        if len(extras) != 0:
            return data, extras
        else:
            return data

    dim_index = choice_fn(data)
    cur_target_num = round(len_data * cur_target_num_scaler)
    if cur_target_num < target_num:
        cur_target_num = target_num

    start_point = np.random.randint(len_data - cur_target_num + 1)
    end_points = start_point + cur_target_num - 1
    start_value = torch.from_numpy(data[:, dim_index]).kthvalue(start_point + 1).values.numpy()
    end_value = torch.from_numpy(data[:, dim_index]).kthvalue(end_points + 1).values.numpy()
    mask = np.logical_and(data[:, dim_index] >= start_value, data[:, dim_index] <= end_value)

    data = data[mask]
    extras = tuple(extra[mask] if isinstance(extra, np.ndarray) else extra for extra in extras)

    if cur_target_num <= target_num:
        if len(extras) != 0:
            return data, extras
        else:
            return data
    return kd_tree_partition_randomly(
        data, target_num, extras, choice_fn
    )


def kd_tree_partition_base(data: np.ndarray, max_num: int) -> List[np.ndarray]:
    if len(data) <= max_num:
        return [data]

    dim_index = np.argmax(np.var(data, 0)).item()
    split_point = len(data) // 2
    split_value = torch.from_numpy(data[:, dim_index]).kthvalue(split_point).values.numpy()
    mask = data[:, dim_index] <= split_value

    if split_point <= max_num:
        return [data[mask], data[~mask]]
    else:
        left_partitions = kd_tree_partition_base(data[mask], max_num)
        right_partitions = kd_tree_partition_base(data[~mask], max_num)
        left_partitions.extend(right_partitions)

    return left_partitions


def kd_tree_partition_extended(data: np.ndarray, max_num: int, extras: List[np.ndarray]) \
        -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    if len(data) <= max_num:
        return [data], [[extra] for extra in extras]

    dim_index = np.argmax(np.var(data, 0)).item()
    split_point = len(data) // 2
    split_value = torch.from_numpy(data[:, dim_index]).kthvalue(split_point).values.numpy()
    mask = data[:, dim_index] <= split_value

    if split_point <= max_num:
        return [data[mask], data[~mask]], [[extra[mask], extra[~mask]] for extra in extras]
    else:
        left_partitions, left_extra_partitions = kd_tree_partition_extended(
            data[mask], max_num,
            [extra[mask] if extra is not None else extra for extra in extras]
        )
        mask = np.logical_not(mask)
        right_partitions, right_extra_partitions = kd_tree_partition_extended(
            data[mask], max_num,
            [extra[mask] if extra is not None else extra for extra in extras]
        )
        left_partitions.extend(right_partitions)
        for idx, p in enumerate(right_extra_partitions):
            left_extra_partitions[idx].extend(p)

        return left_partitions, left_extra_partitions

def kd_tree_partition(data: Union[np.ndarray, torch.Tensor], max_num: int,
                      extras: Union[List[np.ndarray], List[torch.Tensor]] = None)\
        -> Union[List[np.ndarray], List[torch.Tensor],
                 Tuple[List[np.ndarray], List[List[np.ndarray]]],
                 Tuple[List[torch.Tensor], List[List[torch.Tensor]]]]:
    is_torch_tensor = isinstance(data, torch.Tensor)
    if extras is None or extras == []:
        if is_torch_tensor:
            data = data.numpy()
        data_list = kd_tree_partition_base(data, max_num)
        if is_torch_tensor:
            data_list = [torch.from_numpy(_) for _ in data_list]
        return data_list
    else:
        if is_torch_tensor:
            data = data.numpy()
            extras = [_.numpy() for _ in extras]
        data_list, extras_lists = kd_tree_partition_extended(data, max_num, extras)
        if is_torch_tensor:
            data_list = [torch.from_numpy(_) for _ in data_list]
            extras_lists = [[torch.from_numpy(_) for _ in extras_list]
                            for extras_list in extras_lists]
        return data_list, extras_lists