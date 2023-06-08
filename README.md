# [CVPR 2023]Prototype-based Embedding Network for Scene Graph Generation

This repository contains the official code implementation for the paper [Prototype-based Embedding Network for Scene Graph Generation](https://arxiv.org/abs/2303.07096).

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Train
We provide [scripts](./scripts/train.sh) for training the models
```
export CUDA_VISIBLE_DEVICES=1
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

```

## Test
We provide [scripts](./scripts/test.sh) for testing the models
```
export CUDA_VISIBLE_DEVICES=3
export NUM_GUP=1
echo "Testing!!!!"
MODEL_NAME="PE-NET_PredCls"
python3 \
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
```


## Device

All our experiments are conducted on one NVIDIA GeForce RTX 3090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).



## The Trained Model Weights

We provide the weights for the  model. Due to random seeds and machines, they are not completely consistent with those reported in the paper, but they are within the allowable error range.

|      Model       | R@20  | R@50  | R@100 | mR@20 | mR@50 | mR@100 |                         Google Drive                         |
| :--------------: | :---: | :---: | :---: | :---: | :---: | :----: | :----------------------------------------------------------: |
| PE-Net (PredCls) | 58.21 | 65.23 | 67.34 | 25.83 | 31.42 | 33.48  | [Model Link](https://drive.google.com/file/d/1rjsLs3N33iiOB5xYO7zetNhR7ebi385W/view?usp=share_link) \| [Log Link](https://drive.google.com/file/d/1YK0dLWVkmfWQjpreBdeWi4H0XyV61XMl/view?usp=share_link) |
|  PE-Net (SGCls)  | 35.28 | 39.13 | 40.19 | 15.23 | 18.22 | 19.34  | [Model Link](https://drive.google.com/file/d/1uRl-O-yXmpCs__l_V-WTdYtbWPl57M1B/view?usp=share_link) \| [Log Link](https://drive.google.com/file/d/13mjhmEEvQtwHeinrrbgIJFgDQSuryvCL/view?usp=share_link) |
|  PE-Net (SGDet)  | 23.36 | 30.41 | 34.84 | 9.16  | 12.25 | 14.34  | [Model Link](https://drive.google.com/file/d/1Ed6PkATiig0xpFuQYL-G5trFifhPpc0C/view?usp=share_link) \| [Log Link](https://drive.google.com/file/d/1cweQarIDB0PWCA3Xt94-_AGItdw6EQ5h/view?usp=share_link) |

## Tips

We use the `rel_nms` [operation](./maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) provided by [RU-Net](https://github.com/siml3/RU-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) and [HL-Net](https://github.com/siml3/HL-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) in PredCls and SGCls to filter the predicted relation predicates, which encourages diverse prediction results. 

## Help

Be free to contact me (`zheng_chaofan@foxmail.com`) if you have any questions!

## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Citation

```
@inproceedings{zheng2023prototype,
  title={Prototype-based Embedding Network for Scene Graph Generation},
  author={Zheng, Chaofan and Lyu, Xinyu and Gao, Lianli and Dai, Bo and Song, Jingkuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22783--22792},
  year={2023}
}
```
