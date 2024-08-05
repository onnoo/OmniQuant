
MODEL_PATH='/data/hf_models/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/'

CUDA_VISIBLE_DEVICES=0 python main.py \
--model $MODEL_PATH --eval_ppl \
--epochs 1 --output_dir ./log/Llama-2-7b-w6a6 \
--wbits 6 --abits 6 --lwc --let --net Llama-2-7b \
--deactive_amp --save_dir ./log/Llama-2-7b-w6a6 \
--pretrained meta-llama/Llama-2-7b-hf --except_layer