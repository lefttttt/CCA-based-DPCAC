import subprocess
import os


def compute_metrics(gt_file, rec_file, resolution, normal=False, attr=True, job_id=None, return_dict=None):
    """Compute D1 and/or D2 with pc_error tool from MPEG"""

    pc_error_path = '/path/of/pc_error' # Please get pc_error and update the path here
    cmd = pc_error_path + \
        ' --dropdups=2 --averageNormals=1 --neighborsProc=1' + \
        ' --fileA='+ gt_file + \
        ' --fileB='+ rec_file + \
        ' --resolution=' + str(resolution)
    if normal:
        cmd += ' --inputNorm=' + gt_file
    if attr:
        if resolution == 30000: # for 18-bit LiDAR case
            cmd += ' --lidar=1'
        else: # XR
            cmd += ' --color=1'
    bg_proc=subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    line_b = bg_proc.stdout.readline()

    psnr_d1, psnr_d2, number_points_input, number_points_output, psnr_attr = None, None, None, None, []
    d1_key = 'mseF,PSNR (p2point):'
    d2_key = 'mseF,PSNR (p2plane):'
    if attr:
        colour_keys = ['  c[0],PSNRF         :', '  c[1],PSNRF         :', '  c[2],PSNRF         :']
        reflectance_key = 'r,PSNR   F         :'
    num_point_key = 'Point cloud sizes'
    while line_b:
        line = line_b.decode(encoding='utf-8')
        line_b = bg_proc.stdout.readline()
        idx = line.find(d1_key)
        if idx >= 0: psnr_d1 = float(line[idx + len(d1_key):])
        if normal:
            idx = line.find(d2_key)
            if idx >= 0: psnr_d2 = float(line[idx + len(d2_key):])
        if attr:
            for key in colour_keys:
                idx = line.find(key)
                if idx >=0: psnr_attr += [float(line[idx + len(key):])]
            idx = line.find(reflectance_key)
            if idx >= 0: psnr_attr += [float(line[idx + len(reflectance_key):])]
        idx = line.find(num_point_key)
        if idx >= 0:
            number_points_input, number_points_output, _ = line.split(':')[1].split(',')
            number_points_input = int(number_points_input)
            number_points_output = int(number_points_output)
    if len(psnr_attr) == 0: psnr_attr = None

    result = {"psnr_d1": psnr_d1, 
        "psnr_d2": psnr_d2,
        "number_points_input": number_points_input,
        "number_points_output": number_points_output,
        "psnr_attr": psnr_attr}

    if return_dict is not None:
        return_dict[job_id] = result
    else:
        return result
