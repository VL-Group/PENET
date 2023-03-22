export PYTHONPATH=/mnt/hdd1/zhanghaonan/code/code_sgg/lib/apex:/mnt/hdd1/zhanghaonan/code/code_sgg/lib/cocoapi:/mnt/hdd1/zhanghaonan/code/code_sgg/PE-Net/Scene-Graph-Benchmark.pytorch-master:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=6
export NUM_GUP=1
echo "TRAINING Predcls"

MODEL_NAME='PE-NET_PredCls'
mkdir ./checkpoints/${MODEL_NAME}/
cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py ./checkpoints/${MODEL_NAME}/
cp ./scripts/train.sh ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/relation_head.py ./checkpoints/${MODEL_NAME}

python3 \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 30000 \
  SOLVER.CHECKPOINT_PERIOD 30000 GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0;
