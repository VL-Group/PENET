#!/usr/bin/env bash
export PYTHONPATH=/mnt/hdd1/zhanghaonan/code/code_sgg/lib/apex:/mnt/hdd1/zhanghaonan/code/code_sgg/lib/cocoapi:/mnt/hdd1/zhanghaonan/code/code_sgg/Scene-Graph-Benchmark.pytorch-master:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=3
export NUM_GUP=1
echo "Testing!!!!"
MODEL_NAME="PE-NET_PredCls"
python \
        tools/relation_test_net.py \
        --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
        TEST.IMS_PER_BATCH 1 DTYPE "float32" \
        GLOVE_DIR ./datasets/vg/ \
        MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
        MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_final.pth \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
        TEST.ALLOW_LOAD_FROM_CACHE False \