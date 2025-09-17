# Low-Delay Dynamic Point Cloud Attribute Compression via Cross-Coordinate Attention

A dynamic point cloud attribute compression algorithm based on Cross-Coordinate Attention with low-delay.

## Requirments

### Environment
- python == 3.8
- pytorch == 1.12.1
- cuda == 11.3
- MinkowskiEngine == 0.5.4

### Third-party software

Please install `KNN_CUDA/` and `pc_error` in `model/`:
- The github URL of `pc_error` is: https://github.com/MPEGGroup/mpeg-pcc-dmetric.git
- The github URL of `KNN_CUDA` is: https://github.com/unlimblue/KNN_CUDA.git
- Please follow their `README.md` to complete the installation.
- Please update the path of pc_error in pc_error.py file.

## Usage

### Dataset

Please put the dataset in `data/`:
- It is recommended to place various datasets in different folders and distinguish between training datasets and test datasets.
- Modify the corresponding dataset path in the main programs of `train_dynamic.py`, `test_dynamic.py`, `train_static.py` and `test_dynamic.py`.

### Checkpoint

The weight files obtained from training can be placed in `checkpoints/`. Note that the checkpoint paths for training and testing are the same.

### Training

Execute `train_dynamic/static.py` to start training:
- Adjust `batch_size` and `num_points` during training to prevent out of memory.
- Modify `data_path` to point to your dataset path.
- Set `checkpoint_name` to determine where the model weights are saved.
- Set `device_id` to your available GPU ID.

### Testing

Execute `test_dynamic/test.py` to start testing:
- Set `data_path` and `results_path` to point to the path of the test dataset and the storage path of the test results respectively.
- Set `checkpoint_name` to point to the model weights used during testing.
- Set `device_id` to your available GPU ID.
