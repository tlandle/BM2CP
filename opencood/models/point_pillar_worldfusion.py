# point_pillar_worldfusion.py
# --------------------------------------------------------
# Hybrid BM2CP × Where2Comm with world‑canvas output
#
# (c) 2025  – feel free to MIT‑license

from __future__ import annotations
import torch, torch.nn as nn
from einops import rearrange
from typing import Dict, Tuple, List


import numpy as np
from opencood.utils import box_utils, transformation_utils
from opencood.data_utils.pre_processor import SpVoxelPreProcessor
from opencood.data_utils.post_processor import VoxelPostProcessor
from opencood.models.common_modules import (
    PillarVFE, PointPillarScatter, DownsampleConv, NaiveCompressor
)
from opencood.models.bm2cp_modules.sensor_blocks import ImgCamEncode
from opencood.models.bm2cp_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.bm2cp_modules.attentioncomm import         \
        ScaledDotProductAttention
from opencood.models.where2comm_modules.where2comm_attn import Where2comm


class ImgModalFusion(nn.Module):
    def __init__(self, dim, threshold=0.5):
        super().__init__()
        self.att = ScaledDotProductAttention(dim)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.Sigmoid()
        self.thres = threshold               

    def forward(self, img_voxel, pc_voxel):
        B, C, imZ, imH, imW = pc_voxel.shape
        pc_voxel = pc_voxel.view(B, C, -1)
        img_voxel = img_voxel.view(B, C, -1)
        voxel_mask = self.att(pc_voxel, img_voxel, img_voxel)
        voxel_mask = self.act(self.proj(voxel_mask.permute(0,2,1)))
        voxel_mask = voxel_mask.permute(0,2,1)
        voxel_mask = voxel_mask.view(B, C, imZ, imH, imW)

        ones_mask = torch.ones_like(voxel_mask).to(voxel_mask.device)
        zeros_mask = torch.zeros_like(voxel_mask).to(voxel_mask.device)
        mask = torch.where(voxel_mask>self.thres, ones_mask, zeros_mask)

        mask[0] = ones_mask[0]
        
        img_voxel = img_voxel.view(B, C, imZ, imH, imW)
        return mask


class MultiModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.img_fusion = ImgModalFusion(dim)

        self.multigate = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.multifuse = nn.Conv3d(dim*2, dim, 1, 1, 0)

    def forward(self, img_voxel, pc_dict):
        pc_voxel = pc_dict['spatial_features_3d']
        B, C, Z, Y, X = pc_voxel.shape
        print(f"[MultiModalFusion] pc_voxel shape: {pc_voxel.shape}, img_voxel shape: {img_voxel.shape}")
        print(f"[MultiModalFusion] pc_voxel device: {pc_voxel.device}, img_voxel device: {img_voxel.device}")

        # pc->pc; img->img*mask; pc+img->
        ones_mask = torch.ones_like(pc_voxel).to(pc_voxel.device)
        zeros_mask = torch.zeros_like(pc_voxel).to(pc_voxel.device)
        mask = torch.ones_like(pc_voxel).to(pc_voxel.device)

        print(f"[MultiModalFusion] ones_mask shape: {ones_mask.shape}, zeros_mask shape: {zeros_mask.shape}")
        print(f"[MultiModalFusion] ones_mask device: {ones_mask.device}, zeros_mask device: {zeros_mask.device}")

        
        pc_mask = torch.where(pc_voxel!=0, ones_mask, zeros_mask)
        pc_mask, _ = torch.max(pc_mask, dim=1)
        pc_mask = pc_mask.unsqueeze(1)
        img_mask = torch.where(img_voxel!=0, ones_mask, zeros_mask)
        img_mask, _ = torch.max(img_mask, dim=1)
        img_mask = img_mask.unsqueeze(1)

        fused_voxel = pc_mask*img_mask*self.multifuse(torch.cat([self.act(self.multigate(pc_voxel))*img_voxel, pc_voxel], dim=1))
        fused_voxel = fused_voxel + pc_voxel*pc_mask*(1-img_mask) + img_voxel*self.img_fusion(img_voxel, pc_voxel)*(1-pc_mask)*img_mask

        print(f"[MultiModalFusion] fused_voxel shape: {fused_voxel.shape}")
        print(f"[MultiModalFusion] pc_mask shape: {pc_mask.shape}, img_mask shape: {img_mask.shape}")

        thres_map = pc_mask*img_mask*0 + pc_mask*(1-img_mask)*0.5 + (1-pc_mask)*img_mask*0.5 + (1-pc_mask)*(1-img_mask)*0.5
        mask = pc_mask*img_mask + pc_mask*(1-img_mask)*2 + (1-pc_mask)*img_mask*3 + (1-pc_mask)*(1-img_mask)*4
        mask1 = pc_mask
        mask2 = img_mask
        # size = [B, 1, Z, Y, X]
        thres_map, _ = torch.min(thres_map, dim=2)  # collapse Z-axis, dim=4 size = [B, 1, Y, X]
        mask1, _ = torch.max(mask1, dim=2)  # collapse Z-axis, dim=4 size = [B, 1, Y, X]
        mask2, _ = torch.max(mask2, dim=2)  # collapse Z-axis, dim=4 size = [B, 1, Y, X]
        
        pc_dict['spatial_features'] = fused_voxel.view(B,C*Z, Y, X)
        return pc_dict, thres_map, torch.min(mask, dim=2)[0], torch.stack([mask1, mask2])

# ---------------------------------------------------------------------------
# (A)  Sensor‑level encoders  (IDENTICAL to BM2CP)
# ---------------------------------------------------------------------------
class _SensorEncoder(nn.Module):
    """
    ① LiDAR pillar → scatter       (unchanged)
    ② 4‑camera ImgCamEncode        (unchanged)
    ③ Voxel‑space img✕lidar fusion (unchanged)
    ④ ResNet BEV backbone          (unchanged)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg   = cfg
        pc_cfg     = cfg['pc_params']
        img_cfg    = cfg['img_params']
        fuse_cfg   = cfg['modality_fusion']

        # LiDAR branch
        self.pillar_vfe = PillarVFE(pc_cfg['pillar_vfe'],
                                    4, pc_cfg['voxel_size'],
                                    pc_cfg['lidar_range'])
        self.scatter    = PointPillarScatter(pc_cfg['point_pillar_scatter'])

        # Camera branch
        self.camenc     = ImgCamEncode(
            D            = img_cfg['grid_conf']['ddiscr'][-1],
            bev_dim      = img_cfg['bev_dim'],
            img_downsample = img_cfg['img_downsample'],
            ddiscr         = img_cfg['grid_conf']['ddiscr'],
            dmode          = img_cfg['grid_conf']['mode'],
            use_depth_gt   = img_cfg['use_depth_gt'],
            depth_sup      = img_cfg['depth_supervision']
        )
        self.get_geometry = self.camenc.get_geometry  # helper

        # Sensor‑level fusion in voxel space
        self.vox_fuse = MultiModalFusion(img_cfg['bev_dim'])

        # BEV backbone (shared with Where2Comm)
        self.backbone = ResNetBEVBackbone(fuse_cfg['bev_backbone'],
                                          input_channels = img_cfg['bev_dim'])

        # optional mapper to 256‑C / 50×176
        self.shrink_flag = 'shrink_header' in fuse_cfg
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(fuse_cfg['shrink_header'])

        # heads – *local­‑only*, used by Where2Comm to decide communication
        C_out = fuse_cfg['shrink_header']['dim'][0] if self.shrink_flag else \
                sum(fuse_cfg['bev_backbone']['num_upsample_filter'])
        self.cls_head = nn.Conv2d(C_out, cfg['anchor_number'], 1)
        self.reg_head = nn.Conv2d(C_out, 7 * cfg['anchor_number'], 1)

    @torch.inference_mode()
    def forward(self, batch: Dict) -> Dict:
        """Return a **feature_dict** exactly like BM2CP.get_feature()."""

        # -------------- LiDAR ------------------------------------------------
        pc = batch['processed_lidar']
        rec_len = batch['record_len']
        bd = {'voxel_features': pc['voxel_features'],
              'voxel_coords'  : pc['voxel_coords'],
              'voxel_num_points': pc['voxel_num_points'],
              'record_len'      : rec_len}
        bd = self.scatter(self.pillar_vfe(bd))             # + bd['spatial_features_3d']

        # -------------- Cameras ---------------------------------------------
        imgs = batch['image_inputs']['imgs']      # B × N × 3 × H × W
        geom = self.get_geometry(batch['image_inputs'])
        B,N,C,imH,imW = imgs.shape
        imgs = imgs.view(B*N, C, imH, imW)
        _, camv = self.camenc(imgs,
                              batch['image_inputs']['depth_map'],
                              rec_len)
        camv = rearrange(camv, '(b n) c d h w -> b n c d h w', b=B, n=N)
        camv = camv.permute(0,1,3,4,5,2)                # B N D H W C

        # -------------- Voxel‑space fusion ----------------------------------
        bd, thres_map, mask, each_mask = self.vox_fuse(camv, bd)
        bd = self.backbone(bd)

        bev2d = bd['spatial_features_2d']                 # B*C 50 176
        if self.shrink_flag: bev2d = self.shrink_conv(bev2d)

        bd.update({
            'spatial_features_2d': bev2d,            # [∑CAV,256,50,176]
            'psm' : self.cls_head(bev2d),
            'rm'  : self.reg_head(bev2d),
            'thres_map': thres_map,
            'mask'     : mask,
            'each_mask': each_mask
        })
        return bd


# ---------------------------------------------------------------------------
# (B)  Edge‑side cooperative fusion  (Where2Comm)
# ---------------------------------------------------------------------------
class PointPillarWorldFusion(nn.Module):
    """
    – embed the _SensorEncoder as .sensor  
    – reuse Where2Comm (attention‑mask) for latency‑aware feature selection  
    – finally map fused BEV into a *world raster* (200×200 m) for every frame.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg      = cfg
        self.sensor   = _SensorEncoder(cfg)                  # part (A)
        self.fusion   = Where2comm(cfg['fusion_args'])       # attn comm
        self.shrink   = self.sensor.shrink_conv              # reuse
        self.cls_head = self.sensor.cls_head
        self.reg_head = self.sensor.reg_head
        self.multi_scale = cfg['fusion_args']['multi_scale']

        # canvas: 400×400 px @ 0.5 m/px  ⇒  200 m square
        self.canvas_res = 0.5        # metres per pixel
        self.canvas_W = int(200 / self.canvas_res)
        self.canvas_H = int(200 / self.canvas_res)
        self.register_buffer('origin_xy', torch.tensor([0., 0.]))  # RSU‑centre

    # ------------ vehicle side ------------------------------------------------
    @torch.inference_mode()
    def get_feature(self, batch: Dict) -> Dict:
        """wrap sensor encoder so vehicles can call manager.model.get_feature"""
        return self.sensor(batch)

    # ------------ edge side ---------------------------------------------------
    def forward(self,
                feat_list : List[Dict],
                pairwise  : torch.Tensor,
                record_len: torch.Tensor) -> Dict:
        """
        feat_list – list of *feature_dict* from ≥1 agents (order ego, RSUs…)
        pairwise  – [1,L,L,4,4]  transform matrix
        """
        # stack features (exactly like your current edge code)
        spatial = torch.cat([d['spatial_features_2d'] for d in feat_list], 0)
        psm     = torch.cat([d['psm']               for d in feat_list], 0)
        thres   = torch.cat([d['thres_map']         for d in feat_list], 0)

        fused, comm_rate, _ = self.fusion(
              spatial, psm, thres, record_len, pairwise,
              backbone = self.sensor.backbone,
              heads    = [self.shrink, self.cls_head, self.reg_head])

        if self.sensor.shrink_flag:
            fused = self.shrink(fused)
        psm_fused = self.cls_head(fused)
        rm_fused  = self.reg_head(fused)

        # ------------------------------------------------------------------
        # (C) world raster head
        # ------------------------------------------------------------------
        detections, scores = self._decode_boxes(psm_fused, rm_fused)

        canvas = self._paint_canvas(detections)   # 1×1×H×W float mask
        return {'dets': detections, 'scores': scores,
                'canvas': canvas, 'comm_rate': comm_rate}

    # -------- helper: decode anchor boxes (reuse post‑processor) --------------
    def _decode_boxes(self, psm: torch.Tensor, rm: torch.Tensor):
        post = self.postprocessor
        anc  = post.generate_anchor_box()
        data = {'ego': {'anchor_box': torch.from_numpy(anc).to(psm.device),
                        'transformation_matrix': torch.eye(4).to(psm.device)}}
        pred = {'ego': {'psm': psm, 'rm': rm}}
        corners, scores = post.post_process(data, pred)
        if corners is None:                       # nothing
            return torch.empty(0,8,3), torch.empty(0)
        return corners, scores                   # still in anchor‑frame

    # -------- helper: draw onto global canvas ---------------------------------
    def _paint_canvas(self, boxes_h8: torch.Tensor) -> torch.Tensor:
        """
        Very small stand‑in rasteriser.  Marks every box centre (+yaw arrow)
        into a global 400×400 canvas stored as float32.
        """
        canvas = torch.zeros(1, 1, self.canvas_H, self.canvas_W,
                             device=boxes_h8.device)
        if boxes_h8.numel() == 0: return canvas
        ctr = box_utils.corner_to_center(boxes_h8.cpu().numpy(), order='hwl')
        # ctr ­→ world xy (already in RSU world because anchor==RSU)
        xs = ((ctr[:,0] - self.origin_xy[0]) / self.canvas_res + self.canvas_W/2).astype(int)
        ys = ((-ctr[:,1]+ self.origin_xy[1]) / self.canvas_res + self.canvas_H/2).astype(int)
        xs = np.clip(xs, 0, self.canvas_W-1)
        ys = np.clip(ys, 0, self.canvas_H-1)
        canvas[0,0, ys, xs] = 1.            # mark centres
        return canvas
