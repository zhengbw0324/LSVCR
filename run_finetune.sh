export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export DISABLE_MLFLOW_INTEGRATION=TRUE
export TORCH_DISTRIBUTED_DEBUG=DETAIL

SEED=2024
EPOCHS=50
DEV_BATCH_SIZE=512
GRAD_ACCUMULARION_STEPS=1
NUM_WORKERS=12
WARMUP=0.0
EVAL_STEPs=1
WEIGHT_DECAY=1e-5
SCHEDULER_TYPE=constant
DATASET_PATH=./data/
HIS_LEN=50
TEST_CAND_NUM=100

LR=1e-3
NEG=19
W=0.1
ACCELERATE=./config/accelerate_config.yaml


PEMB=photo_embs.npy
CEMB=comment_embs.npy
PRETRAIN_CKPT=./ckpt/LSVCR/adapter_model.bin

TASK=CommRank
#TASK=Rec
RUN_NAME=FtLSVCR
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=./ckpt/finetune/${RUN_NAME}-${TASK}-${DATESTR}-${LR}

accelerate launch --config_file $ACCELERATE finetune.py \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --pretrain_checkpoint $PRETRAIN_CKPT \
    --data_path $DATASET_PATH \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --per_device_eval_batch_size $DEV_BATCH_SIZE \
    --dataloader_num_workers $NUM_WORKERS \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --lr_scheduler_type $SCHEDULER_TYPE \
    --num_train_epochs $EPOCHS \
    --warmup_ratio $WARMUP \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEPs \
    --finetune_task $TASK \
    --max_candidate_num $TEST_CAND_NUM \
    --max_position $HIS_LEN \
    --max_phis_len $HIS_LEN \
    --max_chis_len $HIS_LEN \
    --id_text_loss_weight $W \
    --photo_emb_file $PEMB \
    --comment_emb_file $CEMB \
    --neg_comment_num $NEG