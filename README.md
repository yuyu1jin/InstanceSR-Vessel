# InstanceSR-Vessel

InstanceSR-Vessel is a standalone instance-level super-resolution reconstruction module for UAV-based vessel targets. It contains the training, inference, and independent evaluation code for the reconstruction module only. This project can be integrated as a plug-in module for enhancing general small-object detection.

Image data is not included for copyright reasons. Please prepare your own dataset following the directory structure below before training, evaluation, or inference.

## Ascend NPU Environment
This project was developed and tested in Huawei Ascend NPU hardware based on CANN8.0.RC2

**Hardware**

```text
Huawei Ascend 910B
CANN:      8.0.RC2
PyTorch:   2.3.1
Backend:   Ascend Extension for PyTorch / torch_npu
Device:    npu
```

Ascend Extension for PyTorch, also known as `torch_npu`, adapts PyTorch workloads to Ascend AI Processors. Run `pip install torch-npu==2.3.1` to install this extension.

Recommended runtime checks:

```bash
python -c "import torch; import torch_npu; print(torch.__version__); print(torch.npu.is_available())"
npu-smi info
```

If the process falls back to CPU, check whether the container has access to the Ascend device, whether CANN environment variables are loaded, and whether `torch_npu` matches the installed PyTorch/CANN version.

For more detail, see the following resources:

- Ascend Extension for PyTorch documentation: https://www.hiascend.com/document/detail/zh/Pytorch/
- Ascend PyTorch adapter (`torch_npu`): https://github.com/Ascend/pytorch
- `torch-npu` package page: https://pypi.org/project/torch-npu/

## Project Structure

```text
InstanceSR-Vessel/
├── configs/
│   └── paths.yaml                     --- Path configuration
├── model/                             --- Network definitions
│   ├── generator.py
│   ├── discriminator.py
│   ├── gan_model.py
│   ├── resnet50.py
│   └── stage2_head.py
├── loss/                              --- Training losses
│   ├── Loss.py
│   └── discriminator_loss.py
├── utils/                             --- Common utilities
│   ├── paths.py
│   └── resize_pad.py
├── model_data/                        --- Backbone checkpoints
├── checkpoints/                       --- Released or pretrained module checkpoints
├── datasets/                          --- Training dataset root
│   ├── HR/<class_name>/
│   ├── LR/<class_name>/
│   └── META/resize_meta_<class_name>.csv
├── inference/
│   ├── inputs/                        --- Instance images for inference
│   └── outputs/                       --- Inference outputs
├── Experiments/
│   └── Part_C/
│       ├── testset_instance/          --- Independent instance-level evaluation set
│       │   ├── <class_name>/
│       │   │   ├── img_instance/
│       │   │   └── resize_meta.csv
│       └── results/                   --- Evaluation outputs
├── logger/                            --- Training logs
├── train.py                           --- Training entry point
├── infer.py                           --- Instance inference entry point
├── partc_module.py                    --- Independent evaluation entry point
├── downsample.py                      --- Bicubic downsampling helper
└── readme.md
```

## Path Configuration

The project reads paths from `configs/paths.yaml`.

```yaml
project_root: .

weights:
  stage2_weights: checkpoints/gan_epoch50.pth(example)
  resnet50: model_data/resnet50-19c8e357.pth

datasets:
  HR: datasets/HR
  LR: datasets/LR
  META: datasets/META

train:
  log_dir: logger
  log_csv: logger/gan_log.csv
  loss_curve: logger/gan_loss_curve.png
  checkpoint_dir: logger/weights

inference:
  input_dir: inference/inputs
  output_dir: inference/outputs

experiments:
  partc_testset: Experiments/Part_C/testset_instance
  partc_output: Experiments/Part_C/results
```

## Dataset Layout

Training data should be organized by class:

```text
datasets/HR/<class_name>/      High-resolution instance crops
datasets/LR/<class_name>/      Low-resolution instance crops
datasets/META/                 Metadata CSV files for training
```

Training metadata files:

```text
datasets/META/resize_meta_<class_name>.csv
```

Each metadata CSV must contain at least:

```text
image,new_w,new_h
```

## Independent Evaluation Layout

The independent instance-level evaluation set should be placed as:

```text
Experiments/Part_C/testset_instance/<class_name>/img_instance/
Experiments/Part_C/testset_instance/<class_name>/resize_meta.csv
```

Example:

```text
Experiments/Part_C/testset_instance/boat_class1/img_instance/*.png
Experiments/Part_C/testset_instance/boat_class1/resize_meta.csv
```

## Inference Layout

```text
inference/inputs/       Put instance images here for inference
inference/outputs/      Inference results will be saved here
```

## Training

```bash
python train.py
```

Training logs and loss curves are saved to `logger/`. The checkpoint is saved to `logger/weights/` per five epoch. The best checkpoint move to `checkpoints/` for inference.  

## Instance Inference

```bash
python infer.py
```

The inference script accepts instance images from `inference/inputs/`. Each image is resized and padded to `32x32`, reconstructed to `128x128`, and processed by the output head for class prediction and width-height refinement.

## Independent Evaluation

```bash
python partc_module.py
```

This evaluates the instance-level module on `Experiments/Part_C/testset_instance` and writes classification/regression metrics to `Experiments/Part_C/results`.
