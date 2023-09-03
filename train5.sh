#!/bin/bash

NUM_NODES=1
NUM_CORES=8
NUM_GPUS=1
JOB_NAME="train_TACO_single_5"

CONDA_HOME=$HOME/miniconda3
CONDA_ENV=torchvision

stdbuf -o0 -e0 sbatch \
        -N $NUM_NODES \
        -c $NUM_CORES \
        --gres=gpu:$NUM_GPUS \
        --job-name $JOB_NAME \
        -o 'slurm_single_5.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

nvidia-smi -L

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_1_2_1_2_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 1 --set_cost_giou 2 \
--bbox_loss_coef 1 --giou_loss_coef 2 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_1_2_1_5_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 1 --set_cost_giou 2 \
--bbox_loss_coef 1 --giou_loss_coef 5 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_1_2_5_2_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 1 --set_cost_giou 2 \
--bbox_loss_coef 5 --giou_loss_coef 2 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_1_2_5_5_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 1 --set_cost_giou 2 \
--bbox_loss_coef 5 --giou_loss_coef 5 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_1_5_1_2_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 1 --set_cost_giou 5 \
--bbox_loss_coef 1 --giou_loss_coef 2 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_1_5_1_5_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 1 --set_cost_giou 5 \
--bbox_loss_coef 1 --giou_loss_coef 5 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_1_5_5_2_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 1 --set_cost_giou 5 \
--bbox_loss_coef 5 --giou_loss_coef 2 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_1_5_5_5_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 1 --set_cost_giou 5 \
--bbox_loss_coef 5 --giou_loss_coef 5 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_5_2_1_2_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 5 --set_cost_giou 2 \
--bbox_loss_coef 1 --giou_loss_coef 2 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_5_2_1_5_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 5 --set_cost_giou 2 \
--bbox_loss_coef 1 --giou_loss_coef 5 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_5_2_5_2_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 5 --set_cost_giou 2 \
--bbox_loss_coef 5 --giou_loss_coef 2 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_5_2_5_5_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 5 --set_cost_giou 2 \
--bbox_loss_coef 5 --giou_loss_coef 5 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_5_5_1_2_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 5 --set_cost_giou 5 \
--bbox_loss_coef 1 --giou_loss_coef 2 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_5_5_1_5_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 5 --set_cost_giou 5 \
--bbox_loss_coef 1 --giou_loss_coef 5 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_5_5_5_2_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 5 --set_cost_giou 5 \
--bbox_loss_coef 5 --giou_loss_coef 2 --eos_coef 0.1

python3 main.py \
--coco_path /datasets/TACO-master/data --dataset_file taco_single --output_dir toy_single_5_5_5_5_5_0.1 \
--resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --lr_drop 75 --toy \
--set_cost_class 5 --set_cost_bbox 5 --set_cost_giou 5 \
--bbox_loss_coef 5 --giou_loss_coef 5 --eos_coef 0.1

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF
