DATA_PATH="data/english.txt"
MODEL_NAME_OR_PATH="gpt2-medium"

GPU_ID=1
LENGTH=32
BATCH_SIZE=128
EVAL_BATCH_SIZE=8
THRESHOLD="0.95"
# MAX_GRAD_NORM=1.0
LR="4e-4"
N_EPOCHS=10
RANDOM_SEED=2023

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --data_path ${DATA_PATH}\
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --max_length ${LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --num_epochs ${N_EPOCHS} \
    --threshold ${THRESHOLD} \
    --learning_rate ${LR} \
    --seed ${RANDOM_SEED}