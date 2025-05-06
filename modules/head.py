# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model head modules
换了检测头
"""

import math
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.yolo.utils.tal import dist2bbox, make_anchors


from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher_crowd

import numpy as np
import time


from ultralytics.nn.modules.head import Segment
from ultralytics.utils.loss import v8OBBLoss

from .block import DFL, Proto
from .conv import Conv,DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_

__all__ = ['Counting', 'Segment','Detect', 'Classify']


# the network frmawork of the Regression branch
class RegressionModel(nn.Module):

    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):

        super(RegressionModel, self).__init__()


        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)

        self.act1 = nn.ReLU()


        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()


        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()


        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    def forward(self, x):
        # 通过第一个卷积层和激活函数
        out = self.conv1(x)
        out = self.act1(out)

        # 通过第二个卷积层和激活函数
        out = self.conv2(out)
        out = self.act2(out)

        # 通过输出卷积层，得到回归的输出
        out = self.output(out)

        # 改变张量的维度顺序：将 `[batch_size, channels, height, width]` 变为 `[batch_size, height, width, channels]`
        out = out.permute(0, 2, 3, 1)

        # 将张量的形状调整为 `[batch_size, num_points, 2]`，其中 `2` 表示 (x, y) 坐标
        return out.contiguous().view(out.shape[0], -1, 2)





# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)




def generate_anchor_points(stride=16, row=3, line=3):
    # 计算每一行（竖直方向）和每一列（水平方向）锚点之间的步长
    row_step = stride / row
    line_step = stride / line

    # 计算每个锚点在网格中的水平偏移量
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    # 计算每个锚点在网格中的垂直偏移量
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    # 使用网格生成所有锚点的偏移
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # 将水平和垂直偏移量堆叠，形成锚点的二维坐标 (x, y)
    anchor_points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

    # 返回生成的锚点坐标
    return anchor_points


def shift(shape, stride, anchor_points):
    # 生成特征图中每个位置的偏移量
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride  # 水平方向的偏移
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride  # 垂直方向的偏移

    # 使用网格生成所有位置的偏移坐标
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # 将水平和垂直的偏移量合并为 (x, y) 坐标
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

    # 获取基础锚点和特征图偏移的数量
    A = anchor_points.shape[0]  # 基础锚点数量
    K = shifts.shape[0]  # 特征图位置的数量

    # 将基础锚点加到每个特征图位置偏移上，生成所有锚点的位置
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    # 返回所有锚点的坐标
    return all_anchor_points




# 定义AnchorPoints类，生成所有金字塔特征层上的参考锚点
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        # 调用父类的初始化方法
        super(AnchorPoints, self).__init__()

        # 设置金字塔特征层的层次，默认为 [3, 4, 5, 6, 7]，对应不同尺度的特征图
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        # 设置特征层对应的步长，默认为 2 的层次幂次方，如 [8, 16, 32, 64, 128]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        # 设置网格的行数和列数，用于生成锚点
        self.row = row
        self.line = line

    # 前向传播方法，用于生成给定图像的所有锚点
    def forward(self, image):
        # 获取输入图像的尺寸，假设输入形状为 [batch_size, channels, height, width]
        image_shape = image.shape[2:]
        # 将图像形状转换为 numpy 数组形式，方便后续计算
        image_shape = np.array(image_shape)
        # 计算每个金字塔特征层的特征图形状
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # 初始化一个空数组，用于存储所有锚点
        all_anchor_points = np.zeros((0, 2)).astype(np.float32)

        # 遍历每个金字塔特征层，生成对应的锚点
        for idx, p in enumerate(self.pyramid_levels):
            # 生成参考锚点
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)
            # 将参考锚点移动到特征图的每个位置上
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            # 将生成的锚点追加到所有锚点中
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)


        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))







# Counting
class Counting(nn.Module):
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        self.backbone = backbone  # 骨干特征提取器

        # 计数相关配置
        self.counting_heads = nn.ModuleDict({
            'regression': RegressionModel(num_features_in=256,  # 坐标回归头
                                          num_anchor_points=row * line),
            'classification': ClassificationModel(num_features_in=256,  # 存在性判断头
                                                  num_classes=1,  # 修改为单类别计数判断
                                                  num_anchor_points=row * line)
        })

        # 锚点生成系统
        self.anchor_system = AnchorPoints(
            pyramid_levels=[3],  # 保持与特征层级一致
            row=row,
            line=line
        )

    def forward(self, samples):
        # 特征提取阶段
        feature_maps = self.backbone(samples)  # 获取骨干网络输出

        # 选择中间特征层（假设索引1对应C3特征）
        counting_feature = feature_maps[1]  # [batch, 256, H, W]

        # 计数头并行计算
        coord_offset = self.counting_heads['regression'](counting_feature) * 100  # 坐标偏移量
        existence_prob = self.counting_heads['classification'](counting_feature)  # 存在概率

        # 生成基础锚点网格
        base_anchors = self.anchor_system(samples)  # [1, num_anchors, 2]
        batch_anchors = base_anchors.repeat(coord_offset.size(0), 1, 1)  # 扩展至batch维度

        # 合成最终预测点
        predicted_points = coord_offset + batch_anchors

        return {
            'point_coordinates': predicted_points,  # 预测点绝对坐标
            'presence_confidences': existence_prob  # 各点存在置信度
        }






class Detect(nn.Module):
    """Detect head"""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.
        Args:
            x (tensor): Input tensor.
        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)



#分割
class Segment(Detect):
    """YOLOv8 Segment head for segmentation models.
    输入和输出说明
输入: 由 neck 生成的特征图 x，形状通常为 (batch_size, channels, height, width)。
输出:
分割掩码: 通过卷积和上采样等操作得到的分割掩码，形状为 (batch_size, nc + 1, height, width)，其中 nc + 1 包含了所有类别的分割结果以及背景分割。
    """
    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        # 初始化分割头的类，设置类数、通道数和其他参数。
        super().__init__(nc, ch)  # 调用 Detect 类的构造函数
        ###### Jiayuan changed self.nm to self.nc
        self.npr = 32  # 中间卷积特征的维度，用于分割头的卷积层
        self.cv1 = Conv(ch[0], self.npr, k=3)  # 创建第一个卷积层，将输入通道转换为 self.npr
        self.upsample = nn.ConvTranspose2d(
            self.npr, self.npr // 2, 2, 2, 0, bias=True
        )  # 上采样层，将特征图放大一倍
        # 使用反卷积层进行上采样，输出通道数为 self.npr//2

        self.cv2 = Conv(self.npr // 2, self.npr // 4, k=3)  # 第二个卷积层，将通道数降为 self.npr//4
        # 最后一层卷积，将通道数转换为 (self.nc + 1)，用于输出分割掩码
        self.cv3 = Conv(self.npr // 4, self.nc + 1)  # +1 是为了增加背景通道

        self.sigmoid = nn.Sigmoid()  # 使用 Sigmoid 激活函数，使输出在 [0, 1] 范围内

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.cv3(self.cv2(self.upsample(self.cv1(x[0])))) # mask protos
        if self.training:
            return p
        return p





class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)
