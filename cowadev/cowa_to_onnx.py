from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
import mmcv
import torch
from torch import nn
from functools import partial
import numpy as np
from mmcv.onnx.symbolic import register_op

cfg = '../cowadev/config/retinanet_r50_fpn_rpn_2x_tencent.py'
ckpt = '../work_dirs/retinanet_r50_fpn_rpn_2x_tencent/epoch_2.pth'
out = '/home/cowa006/gitrepo/CRPilot-1.0/trafficsign/det/traffdet.onnx'
inpt = '../data/tencent/det/val/img/10007_20171116_071955.jpg'
input_shape = (1920, 1080)

inpt = mmcv.imread(inpt)

normalize_cfg = {
    'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
    'std': np.array([58.395, 57.12, 57.375], dtype=np.float32)
}

one_img = mmcv.imnormalize(inpt, normalize_cfg['mean'], normalize_cfg['std'])
one_img = mmcv.imresize(one_img, input_shape).transpose(2, 0, 1)
one_img = torch.from_numpy(one_img).unsqueeze(0).float()

cfg = mmcv.Config.fromfile(cfg)

cfg.model.pretrained = None
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
cfg.data.test.test_mode = True
model.cpu().eval()

(_, C, H, W) = one_img.shape
one_meta = {
    'img_shape': (H, W, C),
    'ori_shape': (H, W, C),
    'pad_shape': (H, W, C),
    'filename': '<demo>.png',
    'scale_factor': 1.0,
    'flip': False
}


def delta2bbox(rois,
               deltas,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 4 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.

    Returns:
        Tensor: Boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.
    """
    xy_mask = torch.tensor([[[1.0, 1.0, 0, 0]]], dtype=deltas.dtype).expand_as(deltas)
    wh_mask = torch.tensor([[[0, 0, 1.0, 1.0]]], dtype=deltas.dtype).expand_as(deltas)
    xy_mask = torch.tensor(xy_mask.numpy())
    wh_mask = torch.tensor(wh_mask.numpy())
    # max_ratio = np.abs(np.log(wh_ratio_clip))
    # dw = dw.clamp(min=-max_ratio, max=max_ratio)
    # dh = dh.clamp(min=-max_ratio, max=max_ratio)

    # Compute center of each roi
    pxy = (rois[:, :, :2] + rois[:, :, 2:]) / 2
    # Compute width/height of each roi
    pwh = rois[:, :, 2:] - rois[:, :, :2]
    pxy = torch.tensor(torch.cat([pxy, pxy], dim=-1).numpy())
    pwh = torch.tensor(torch.cat([pwh, pwh], dim=-1).numpy())

    # Use exp(network energy) to enlarge/shrink each roi
    gwh = pwh * deltas.exp() * wh_mask
    # Use network energy to shift the center of each roi
    gxy = (pxy + pwh * deltas) * xy_mask
    # gx = px + pw * dx
    # gy = py + ph * dy
    # # Convert center-xy/width/height to top-left, bottom-right
    # x1 = gx - gw * 0.5
    # y1 = gy - gh * 0.5
    # x2 = gx + gw * 0.5
    # y2 = gy + gh * 0.5
    # if max_shape is not None:
    #     x1 = x1.clamp(min=0, max=max_shape[1])
    #     y1 = y1.clamp(min=0, max=max_shape[0])
    #     x2 = x2.clamp(min=0, max=max_shape[1])
    #     y2 = y2.clamp(min=0, max=max_shape[0])
    # bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return gwh + gxy


class ExportedMdl(nn.Module):
    def __init__(self, model):
        super(ExportedMdl, self).__init__()
        self.img_pad = nn.ZeroPad2d((0, 0, 4, 4))
        self.model = model

    def forward(self, img):
        img = self.img_pad(img)
        x = self.model.backbone(img)
        x = self.model.neck(x)

        cls_scores, bbox_preds = self.model.bbox_head(x)

        batch_size = int(cls_scores[0].size(0))

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        anchors = self.model.bbox_head.anchor_generator.grid_anchors(featmap_sizes, device='cpu')

        anchors = [a.unsqueeze(0) for a in anchors]

        bbox_out = []
        cls_out = []
        for cls_score, bbox_pred, anchor in zip(cls_scores, bbox_preds, anchors):
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            bbox_out.append(delta2bbox(anchor, bbox_pred, one_meta['img_shape']))
            cls_out.append(cls_score.sigmoid())

        bbox_out = torch.cat(bbox_out, dim=1)
        cls_out = torch.cat(cls_out, dim=1)
        return torch.cat([bbox_out, cls_out], dim=-1)


model = ExportedMdl(model)


# model.forward = partial(model.forward, img_metas=[[one_meta]], return_loss=False)
# ret = model(one_img)
# register_extra_symbolics(9)

def upsample_nearest2d(g, input, output_size, *args):
    # height_scale = float(output_size[-2]) / input.type().sizes()[-2]
    # width_scale = float(output_size[-1]) / input.type().sizes()[-1]
    return g.op("Upsample", input,
                scales_f=(1, 1, 2, 2),
                mode_s="nearest")

register_op('upsample_nearest2d', upsample_nearest2d, '', 9)

torch.onnx.export(
    model, one_img,
    out,
    export_params=True,
    keep_initializers_as_inputs=True,
    verbose=True,
    opset_version=9,
    input_names=['input'],
    output_names=['bbox'],
    enable_onnx_checker=False
)
