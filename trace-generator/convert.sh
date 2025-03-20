#!/bin/bash

for i in {0..31}; do

    python3 /home/llm-sim-project/trace-generator/chakra_convert.py \
        --input_filename="/home/llm-sim-project/multidim_test/trace/rank0 copy $i.json" \
        --output_filename="/home/llm-sim-project/multidim_test/trace/rank.$i.et" \
        --input_type="PyTorch"

done