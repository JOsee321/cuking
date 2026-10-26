#!/usr/bin/env bash
# Run hyperparameter sweep for Cuking on GPU 0
set -e

CUDA_VISIBLE_DEVICES=0 python -m cuking.sweep "$@"
