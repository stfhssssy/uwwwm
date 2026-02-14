#!/bin/bash
# re-validate login information
export DATASET=webarena
mkdir -p ./.auth
python -m browser_env.auto_login