# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter 

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 

from .utils_motifs import to_onehot, encode_box_info
from maskrcnn_benchmark.modeling.make_layers import make_fc




@registry.ROI_RELATION_PREDICTOR.register("PrototypeEmbeddingNetwork")
class PrototypeEmbeddingNetwork(nn.Module):
    def __init__(self, config, in_channels):
        super(PrototypeEmbeddingNetwork, self).__init__()

        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels
        

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048 # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)  

        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2 # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT
        
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
       
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim*2, self.mlp_dim)  
        self.gate_obj = nn.Linear(self.mlp_dim*2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim*2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
       
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2) 

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):

        add_losses = {}
        add_data = {}

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        ##### 

        entity_rep = self.post_emb(roi_features)   # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)    # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)    # xo

        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps, entity_preds, entity_embeds, proposals):

            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)
            
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj)) # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)  
        pair_pred = cat(pair_preds, dim=0) 

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        
        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        ######

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.training:

            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)  
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            ### end
            
            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, 51, -1) 
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(51, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})
            ### end 

            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 51, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), 51).cuda()  
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end 
 
        return entity_dists, rel_dists, add_losses, add_data


    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x
    
    
def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2



@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
                
        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True),])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)
        
        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)
        
        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

        
    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append( head_rep[pair_idx[:,0]] - tail_rep[pair_idx[:,1]] )
            else:
                ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list
        
        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats
        
        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep  
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':   # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE': # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            #union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            #union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            #union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            #union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest
            
        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
