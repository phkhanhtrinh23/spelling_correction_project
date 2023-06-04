DATA_PATH="data/english.txt"
MODEL_NAME_OR_PATH="gpt2-medium"

GPU_ID=0
LENGTH=64
BATCH_SIZE=1
THRESHOLD=0.95
MAX_GRAD_NORM=1.0
LR="1e-3"
N_EPOCHS=15
RANDOM_SEED=2023

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --data_path ${DATA_PATH}\
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --max_length ${LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${N_EPOCHS} \
    --threshold ${THRESHOLD} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --learning_rate ${LR} \
    --seed ${RANDOM_SEED}