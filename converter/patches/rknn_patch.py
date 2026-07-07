import torch
import torch.nn as nn


def apply():
    import ultralytics.nn.modules.head as H

    _patch_detect(H.Detect)
    _patch_obb(H.OBB)
    _patch_pose(H.Pose)
    # _patch_segment(H.Segment)  # TODO: enable for seg testing


def _patch_detect(cls):
    original = cls.forward

    def patched(self, x):
        if (self.export and self.format == "rknn") or getattr(self, '_rknn_mode', False):
            return _rknn_detect(self, x)
        return original(self, x)

    cls.forward = patched


def _rknn_detect(self, x):
    nl = self.nl
    out = []
    for i in range(nl):
        box = self.cv2[i](x[i])
        cls = self.cv3[i](x[i]).sigmoid()
        sm = cls.sum(dim=1, keepdim=True)
        out.extend([box, cls, sm])
    return out


def _patch_obb(cls):
    original = cls.forward

    def patched(self, x):
        if (self.export and self.format == "rknn") or getattr(self, '_rknn_mode', False):
            return _rknn_obb(self, x)
        return original(self, x)

    cls.forward = patched


def _rknn_obb(self, x):
    nl = self.nl
    out = []
    for i in range(nl):
        box = self.cv2[i](x[i])
        cls = self.cv3[i](x[i]).sigmoid()
        angle = self.cv4[i](x[i])
        out.extend([box, cls, angle])
    return out


def _rknn_pose(self, x):
    nl = self.nl
    out = []
    for i in range(nl):
        box = self.cv2[i](x[i])
        cls = self.cv3[i](x[i]).sigmoid()
        kpt = self.cv4[i](x[i])
        out.extend([box, cls, kpt])
    return out


def _patch_pose(cls):
    original = cls.forward

    def patched(self, x):
        if (self.export and self.format == "rknn") or getattr(self, '_rknn_mode', False):
            return _rknn_pose(self, x)
        return original(self, x)

    cls.forward = patched


# def _patch_segment(cls):
#     ...
