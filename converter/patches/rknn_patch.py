import torch
import torch.nn as nn


def apply():
    import ultralytics.nn.modules.head as H

    _patch_detect(H.Detect)
    # _patch_segment(H.Segment)  # TODO: enable for seg testing
    # _patch_pose(H.Pose)        # TODO: enable for pose testing
    # _patch_obb(H.OBB)          # TODO: enable for obb testing


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


# def _patch_segment(cls):
#     ...
#
# def _patch_pose(cls):
#     ...
#
# def _patch_obb(cls):
#     ...
