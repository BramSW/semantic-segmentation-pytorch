#!/usr/bin/env bash

MGPU=0
FILENAME="train_generic.py"

OUTPUT_FOLDER="output/finetune_model"
mkdir -p $OUTPUT_FOLDER

# path to the pretrained weights
# PRETRAINED_MODEL="ae_state_dict.pth"
PRETRAINED_MODEL="classifier.pth"
# Number of epochs is set to 400 so that the learning rate decay scheme does not need to be modified.
# The model converges within the first 150 epochs.

python3 "$FILENAME" \
  --arch_encoder "ptresnet50dilated" \
  --arch_decoder "ppm" \
  --gpus "0-$MGPU" \
  --image_mode "rgb" \
  --weights_encoder "$PRETRAINED_MODEL" \
  --dataset "nyuv2sn40" \
  --random_flip 0 \
  --freeze_until "layer4" \
  --ckpt "$OUTPUT_FOLDER"  \
  --epoch_iters 50 \
  --num_epoch 400
