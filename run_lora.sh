export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export DISABLE_MLFLOW_INTEGRATION=TRUE
export TORCH_DISTRIBUTED_DEBUG=DETAIL


NUM_GPUS=8
MASTER_PORT=13324
LR=3e-4
SEED=2024
EPOCHS=1
DEV_BATCH_SIZE=2
GRAD_ACCUMULARION_STEPS=8
NUM_WORKERS=12

WARMUP=0.03
WEIGHT_DECAY=0.0
STRATEGY=steps
SAVE_EVAL_STEPS=500
DEEPSPEED=./config/ds_z2_bf16.json
GRAD_CKPT=True

BASE_MODEL_PATH=/THUDM/chatglm3-6b/
DATASET_PATH=./data/

RUN_NAME=LSVCR
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=./ckpt/${RUN_NAME}-${DATESTR}-${LR}

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train.py \
    --seed $SEED \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --data_path $DATASET_PATH \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --per_device_eval_batch_size $DEV_BATCH_SIZE \
    --dataloader_num_workers $NUM_WORKERS \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --gradient_checkpointing $GRAD_CKPT \
    --learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --warmup_ratio $WARMUP \
    --weight_decay $WEIGHT_DECAY \
    --save_strategy $STRATEGY \
    --bf16 True \
    --lora True \
    --deepspeed $DEEPSPEED \
    --save_steps $SAVE_EVAL_STEPS \
    --load_best_model_at_end False


sleep 180
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 1 --batch_index 0 >batch0.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 1 --batch_index 1 >batch1.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 2 --batch_index 2 >batch2.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 2 --batch_index 3 >batch3.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 3 --batch_index 4 >batch4.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 3 --batch_index 5 >batch5.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 4 --batch_index 6 >batch6.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 4 --batch_index 7 >batch7.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 5 --batch_index 8 >batch8.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 5 --batch_index 9 >batch9.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 6 --batch_index 10 >batch10.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 6 --batch_index 11 >batch11.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 7 --batch_index 12 >batch12.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 7 --batch_index 13 >batch13.log 2>&1 &
sleep 30

nohup python -u photo_title_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 0 >title-emb.log 2>&1 &
sleep 30
nohup python -u comment_emb.py --lora_ckpt $OUTPUT_DIR --gpu_id 0 --batch_index 14 >batch14.log 2>&1 &




