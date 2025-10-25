#!/usr/bin/env bash
set -euo pipefail

INPUT_IMAGE="./data/input/01.png"      
OUTPUT_DIR="./data/output"
ENC_PERIOD=16
SUB=0.005
INDEL=0.00003
SEG="deeplab"   # unet / deeplab / segformer
DEVICE="cpu"    # cuda / cpu

python main.py \
  --input_image "$INPUT_IMAGE" \
  --output_dir "$OUTPUT_DIR" \
  --encoding_period $ENC_PERIOD \
  --sub_ratio $SUB \
  --indel_ratio $INDEL \
  --seg_model $SEG \
  --device $DEVICE
