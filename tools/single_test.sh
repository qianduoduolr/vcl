#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test_prop.py --config $CONFIG --checkpoint $CHECKPOINT
