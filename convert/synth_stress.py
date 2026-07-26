#!/usr/bin/env python3
"""Generate a synthetic ONNX model for NPU stress testing.

Heavy conv-only graph designed to saturate RK3588 NPU (~6 TOPS).
All ops are RKNN-safe: Conv, Relu, BatchNorm.

Usage:
    python3 synth_stress.py                    # → synth_stress.onnx
    python3 synth_stress.py -o /path/out.onnx
    python3 synth_stress.py --layers 12 --channels 256
"""

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def make_conv_bn_relu(graph, name, in_name, out_name, in_ch, out_ch,
                      kernel=3, stride=1, padding=1, weight_init=None):
    """Add Conv + BatchNorm + Relu to graph."""
    w_shape = [out_ch, in_ch, kernel, kernel]
    b_shape = [out_ch]

    w_init = numpy_helper.from_array(
        (weight_init if weight_init is not None else np.random.randn(*w_shape).astype(np.float32) * 0.01),
        name=f"{name}_weight"
    )
    b_init = numpy_helper.from_array(np.zeros(b_shape, dtype=np.float32), name=f"{name}_bias")

    bn_scale = numpy_helper.from_array(np.ones(out_ch, dtype=np.float32), name=f"{name}_bn_scale")
    bn_bias = numpy_helper.from_array(np.zeros(out_ch, dtype=np.float32), name=f"{name}_bn_bias")
    bn_mean = numpy_helper.from_array(np.zeros(out_ch, dtype=np.float32), name=f"{name}_bn_mean")
    bn_var = numpy_helper.from_array(np.ones(out_ch, dtype=np.float32), name=f"{name}_bn_var")

    graph.initializer.extend([w_init, b_init, bn_scale, bn_bias, bn_mean, bn_var])

    conv_out = f"{name}_conv_out"
    graph.node.append(helper.make_node("Conv", [in_name, f"{name}_weight", f"{name}_bias"],
                                        [conv_out], name=f"{name}_conv",
                                        kernel_shape=[kernel, kernel],
                                        strides=[stride, stride],
                                        pads=[padding, padding, padding, padding]))

    bn_out = f"{name}_bn_out"
    graph.node.append(helper.make_node("BatchNormalization",
                                        [conv_out, f"{name}_bn_scale", f"{name}_bn_bias",
                                         f"{name}_bn_mean", f"{name}_bn_var"],
                                        [bn_out], name=f"{name}_bn",
                                        epsilon=1e-5, momentum=0.1))

    graph.node.append(helper.make_node("Relu", [bn_out], [out_name], name=f"{name}_relu"))


def generate_stress_model(layers=10, channels=128, input_size=640, output_path="synth_stress.onnx"):
    """Generate a heavy conv-only ONNX model for NPU stress testing."""
    np.random.seed(42)

    graph = helper.make_graph([], "synth_stress", [], [])

    x0 = "input"
    graph.input.append(helper.make_tensor_value_info(x0, TensorProto.FLOAT, [1, 3, input_size, input_size]))

    cur_in = x0
    cur_ch = 3

    # Build layer configs: expand channels, then contract
    ch_configs = []
    for i in range(layers):
        if i < layers // 2:
            out_ch = min(channels, cur_ch * 2 if cur_ch < channels else channels)
        else:
            out_ch = max(3, cur_ch // 2) if i < layers - 1 else 3
        ch_configs.append(out_ch)
        cur_ch = out_ch

    for i, out_ch in enumerate(ch_configs):
        layer_name = f"layer{i}"
        out_name = f"feat{i}"
        in_ch = 3 if i == 0 else ch_configs[i - 1]
        make_conv_bn_relu(graph, layer_name, cur_in, out_name, in_ch, out_ch)
        cur_in = out_name

    # Output
    graph.output.append(helper.make_tensor_value_info(cur_in, TensorProto.FLOAT, [1, 3, input_size, input_size]))

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    model.producer_name = "synth_stress"
    model.doc_string = "Synthetic conv-only model for RKNN NPU stress testing"

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    total_params = sum(
        np.prod(init.dims) for init in model.graph.initializer
    )
    conv_nodes = sum(1 for n in model.graph.node if n.op_type == "Conv")
    print(f"Model: {layers} layers, channels={channels}, input={input_size}x{input_size}")
    print(f"Conv layers: {conv_nodes}, Total params: {total_params:,.0f}")
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ONNX for NPU stress")
    parser.add_argument("-o", "--output", default="synth_stress.onnx")
    parser.add_argument("--layers", type=int, default=10, help="Number of conv layers")
    parser.add_argument("--channels", type=int, default=128, help="Max channel count")
    parser.add_argument("--input-size", type=int, default=640, help="Spatial input size")
    args = parser.parse_args()

    generate_stress_model(args.layers, args.channels, args.input_size, args.output)


if __name__ == "__main__":
    main()
