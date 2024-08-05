
MODEL_PATH='/data/hf_models/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/'


python main.py \
--model $MODEL_PATH \
--epochs 0 --output_dir ./log/test \
--eval_ppl --wbits 6 --abits 6 --lwc \
--net Llama-2-7b \
--resume ./misc/Llama-2-7b-w6a6.pth
