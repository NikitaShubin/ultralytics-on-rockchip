import torch
import torch.nn as nn


def apply():
    """Применить все патчи RKNN к головам ultralytics."""
    import ultralytics.nn.modules.head as H

    _patch_detect(H.Detect)
    _patch_obb(H.OBB)
    _patch_pose(H.Pose)
    _patch_segment(H.Segment)
    _patch_segment26(H.Segment26)
    _patch_pose26(H.Pose26)
    _patch_obb26(H.OBB26)
    _patch_rknn_config_fuse_bug()
    _patch_fold_constant()


def _patch_rknn_config_fuse_bug():
    """Обход ошибки схождения MatMul в rknn-toolkit2 при GELU активациях.

    Отключает правило fuse_mul_into_gemm, которое вызывает:
        ValueError: operands could not be broadcast together
    когда модель содержит GEMM, за которым следует поэлементное Mul (типично для GELU).
    """
    try:
        from rknn.api import RKNN

        _orig_config = RKNN.config

        def _patched_config(self, **kwargs):
            kwargs.setdefault("disable_rules", [])
            if "fuse_mul_into_gemm" not in kwargs["disable_rules"]:
                kwargs["disable_rules"] = list(kwargs["disable_rules"]) + ["fuse_mul_into_gemm"]
            return _orig_config(self, **kwargs)

        RKNN.config = _patched_config
    except ImportError:
        pass


def _patch_fold_constant():
    """Обход ошибки fold_constant в rknn-toolkit2 для трансформерных моделей.

    Deformable attention RT-DETR создаёт ONNX графы с динамическими формами,
    которые вызывают сбой GraphOptimizer.fold_constant во время onnxruntime.
    Пропуск этого шага позволяет сборке завершиться (операции GridSample
    используют CPU в runtime).
    """
    try:
        from rknn.api.graph_optimizer import GraphOptimizer

        def _patched_fold_constant(self, *args, **kwargs):
            print("[PATCH] пропуск fold_constant (трансформерная модель)")
            return

        GraphOptimizer.fold_constant = _patched_fold_constant
    except ImportError:
        pass


def _patch_detect(cls):
    """Патч головы Detect для совместимости с RKNN."""
    original = cls.forward

    def patched(self, x):
        if (self.export and self.format == "rknn") or getattr(self, '_rknn_mode', False):
            return _rknn_detect(self, x)
        return original(self, x)

    cls.forward = patched


def _rknn_detect(self, x):
    """RKNN совместимый forward для Detect."""
    nl = self.nl
    out = []
    for i in range(nl):
        box = self.cv2[i](x[i])
        cls = self.cv3[i](x[i]).sigmoid()
        sm = cls.sum(dim=1, keepdim=True)
        out.extend([box, cls, sm])
    return out


def _patch_obb(cls):
    """Патч головы OBB для совместимости с RKNN."""
    original = cls.forward

    def patched(self, x):
        if (self.export and self.format == "rknn") or getattr(self, '_rknn_mode', False):
            return _rknn_obb(self, x)
        return original(self, x)

    cls.forward = patched


def _rknn_obb(self, x):
    """RKNN совместимый forward для OBB."""
    nl = self.nl
    out = []
    for i in range(nl):
        box = self.cv2[i](x[i])
        cls = self.cv3[i](x[i]).sigmoid()
        angle = self.cv4[i](x[i])
        out.extend([box, cls, angle])
    return out


def _rknn_pose(self, x):
    """RKNN совместимый forward для Pose."""
    nl = self.nl
    out = []
    has_kpts_head = hasattr(self, 'cv4_kpts') and self.cv4_kpts is not None
    for i in range(nl):
        box = self.cv2[i](x[i])
        cls = self.cv3[i](x[i]).sigmoid()
        feat = self.cv4[i](x[i])
        if has_kpts_head:
            kpt = self.cv4_kpts[i](feat)
        else:
            kpt = feat
        out.extend([box, cls, kpt])
    return out


def _patch_pose(cls):
    """Патч головы Pose для совместимости с RKNN."""
    original = cls.forward

    def patched(self, x):
        if (self.export and self.format == "rknn") or getattr(self, '_rknn_mode', False):
            return _rknn_pose(self, x)
        return original(self, x)

    cls.forward = patched


def _patch_segment(cls):
    """Патч головы Segment для совместимости с RKNN."""
    original = cls.forward

    def patched(self, x):
        if (self.export and self.format == "rknn") or getattr(self, '_rknn_mode', False):
            return _rknn_segment(self, x)
        return original(self, x)

    cls.forward = patched


def _rknn_segment(self, x):
    """RKNN совместимый forward для Segment."""
    nl = self.nl
    out = []
    p = self.proto(x[0])
    for i in range(nl):
        box = self.cv2[i](x[i])
        cls = self.cv3[i](x[i]).sigmoid()
        coeff = self.cv4[i](x[i])
        out.extend([box, cls, coeff])
    out.append(p)
    return out


def _rknn_segment26(self, x):
    """RKNN совместимый forward для Segment26."""
    nl = self.nl
    out = []
    p = self.proto(x)  # Proto26 ожидает полный список признаков backbone
    for i in range(nl):
        box = self.cv2[i](x[i])
        cls = self.cv3[i](x[i]).sigmoid()
        coeff = self.cv4[i](x[i])
        out.extend([box, cls, coeff])
    out.append(p)
    return out


def _patch_segment26(cls):
    """Патч головы Segment26 для совместимости с RKNN."""
    original = cls.forward

    def patched(self, x):
        if (self.export and self.format == "rknn") or getattr(self, '_rknn_mode', False):
            return _rknn_segment26(self, x)
        return original(self, x)

    cls.forward = patched


def _patch_pose26(cls):
    """Патч головы Pose26 для совместимости с RKNN."""
    original = cls.forward

    def patched(self, x):
        if (self.export and self.format == "rknn") or getattr(self, '_rknn_mode', False):
            return _rknn_pose(self, x)
        return original(self, x)

    cls.forward = patched


def _patch_obb26(cls):
    """Патч головы OBB26 для совместимости с RKNN."""
    original = cls.forward

    def patched(self, x):
        if (self.export and self.format == "rknn") or getattr(self, '_rknn_mode', False):
            return _rknn_obb(self, x)
        return original(self, x)

    cls.forward = patched
