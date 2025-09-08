num_gpus=1
per_gpu_batchsize=16

# === VQA-RAD ===
python main.py with data_root=data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_vqa_rad \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=224 \
 test_only=True \
 tokenizer=downloaded/roberta-base \
 load_path=checkpoints/VQARAD_78.27.ckpt

# === SLACK ===
python main.py with data_root=data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_slack \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=224 \
 test_only=True \
 tokenizer=downloaded/roberta-base \
 load_path=checkpoints/SLAK_82.80.ckpt
