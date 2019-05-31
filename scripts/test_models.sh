#!/usr/bin/bash

FILENAME="eval_generic.py"
ARCHNAME_ENCODER="ptresnet50dilated"
ARCHNAME_DECODER="ppm"
IMAGE_MODE="rgb" # either rgb or lab
DATASET="nyuv2sn40"
SPLIT="test"

# folder where the checkpoints are stored
CHECKPOINT_FOLDER="./output/finetune_model"

# single gpu testing
GPU=0

# number of epochs to test
NUM_EPOCHS_TO_TEST=150
SKIP_EPOCH=5

for ((j=0; j<NUM_EPOCHS_TO_TEST; j+=SKIP_EPOCH))
  do
    SUFFIX="_epoch_${j}.pth"
    python3 "$FILENAME" \
     --arch_encoder "$ARCHNAME_ENCODER" \
     --arch_decoder "$ARCHNAME_DECODER" \
     --image_mode "$IMAGE_MODE" \
     --dataset "${DATASET}" \
     --split_name "$SPLIT" \
     --dirname "$CHECKPOINT_FOLDER" \
     --suffix "$SUFFIX" \
     --gpu $GPU
done
