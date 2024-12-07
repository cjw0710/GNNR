GPU=1
DATASET=citeseer
METHOD=rphgnn
USE_NRL=True
TRAIN_STRATEGY=common
USE_INPUT=True
ALL_FEAT=True 
INPUT_DROP_RATE=0.1
DROP_RATE=0.4
HIDDEN_SIZE=64
SQUASH_K=3
EPOCHS=500
MAX_PATIENCE=50
EMBEDDING_SIZE=512
USE_LABEL=False
EVEN_ODD="all"
SEED=2

python -u main_rphgnn.py \
    --dataset ${DATASET} \
    --method ${METHOD} \
    --use_nrl ${USE_NRL} \
    --use_label ${USE_LABEL} \
    --even_odd ${EVEN_ODD} \
    --train_strategy ${TRAIN_STRATEGY} \
    --use_input ${USE_INPUT} \
    --input_drop_rate ${INPUT_DROP_RATE} \
    --drop_rate ${DROP_RATE} \
    --hidden_size ${HIDDEN_SIZE} \
    --squash_k ${SQUASH_K} \
    --num_epochs ${EPOCHS} \
    --max_patience ${MAX_PATIENCE} \
    --embedding_size ${EMBEDDING_SIZE} \
    --use_all_feat ${ALL_FEAT} \
    --output_dir outputs/citeseer/ \
    --gpus ${GPU} \
    --seed ${SEED} > curriculum_citeseer_2.out 2>&1 

python main_rphgnn.py  --dataset "citeseer" --method "rphgnn" --use_nrl True --use_label False --even_odd "all"  --train_strategy "common" --use_input True --input_drop_rate 0.1 --drop_rate 0.4 --hidden_size 512 --squash_k 3 --num_epochs 300 --max_patience 50 --embedding_size 512 --use_all_feat True --output_dir "outputs/citeseer/" --gpus 1 --seed 2 