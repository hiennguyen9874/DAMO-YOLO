#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import argparse
import os

import onnx
import torch
from loguru import logger
from torch import nn

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from damo.base_models.core.end2end import End2End
from damo.base_models.core.ops import RepConv, SiLU
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils.model_utils import get_model_info, replace_module


def make_parser():
    parser = argparse.ArgumentParser("damo converter deployment toolbox")
    # model part
    parser.add_argument(
        "-f",
        "--config_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("--trt", action="store_true", help="whether convert onnx into tensorrt")
    parser.add_argument("--half", action="store_true", help="whether use fp16")
    parser.add_argument("--batch_size", type=int, default=None, help="inference image batch nums")
    parser.add_argument("--img_size", type=int, default="640", help="inference image shape")
    # onnx part
    parser.add_argument("--input", default="images", type=str, help="input node name of onnx model")
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument("-o", "--opset", default=11, type=int, help="onnx opset version")
    parser.add_argument("--trt-version", type=int, default=8, help="tensorrt version")
    parser.add_argument("--end2end", action="store_true", help="export end2end onnx")
    parser.add_argument("--ort", action="store_true", help="export onnx for onnxruntime")
    parser.add_argument("--trt_eval", action="store_true", help="trt evaluation")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--topk-all", default=100, type=int, help="topk")
    parser.add_argument("--iou-thres", default=0.65, type=float, help="iou threshold")
    parser.add_argument("--conf-thres", default=0.25, type=float, help="conf threshold")
    parser.add_argument("--with-preprocess", action="store_true", help="with preprocess")
    parser.add_argument("--concat", action="store_true", help="concat box and score to one output")

    return parser


@logger.catch
def trt_export(onnx_path, batch_size, inference_h, inference_w, half):
    import tensorrt as trt

    if half:
        trt_mode = "fp16"
    else:
        trt_mode = "fp32"

    TRT_LOGGER = trt.Logger()
    engine_path = onnx_path.replace(".onnx", f"_{trt_mode}_bs{batch_size}.trt")

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    logger.info(f"trt_{trt_mode} converting ...")
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        EXPLICIT_BATCH
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = batch_size
        print("Loading ONNX file from path {}...".format(onnx_path))

        if half is True:
            assert builder.platform_has_fast_fp16 == True, "not support fp16"
            builder.fp16_mode = True

        with open(onnx_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        network.get_input(0).shape = [batch_size, 3, inference_h, inference_w]
        print("Completed parsing of ONNX file")
        engine = builder.build_cuda_engine(network)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        logger.info("generated trt engine named {}".format(engine_path))
    return engine_path


@logger.catch
def main():
    args = make_parser().parse_args()

    logger.info("args value: {}".format(args))
    onnx_name = os.path.split(args.config_file)[-1].replace(".py", ".onnx")

    if args.end2end:
        onnx_name = onnx_name.replace(".onnx", "_end2end.onnx")

    # Check device
    cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device}" if cuda else "cpu")
    assert not (
        device.type == "cpu" and args.half
    ), "--half only compatible with GPU export, i.e. use --device 0"
    # init and load model
    config = parse_config(args.config_file)

    if args.batch_size is not None:
        config.test.batch_size = args.batch_size

    # build model
    model = build_local_model(config, "cuda")
    # load model paramerters
    ckpt = torch.load(args.ckpt, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    logger.info("loading checkpoint done.")

    model = replace_module(model, nn.SiLU, SiLU)

    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()

    info = get_model_info(model, (args.img_size, args.img_size))
    logger.info(info)
    # decouple postprocess
    model.head.nms = False

    if not args.end2end and args.concat:
        model.head.concat = True

    if args.end2end:
        model = End2End(
            model,
            max_obj=args.topk_all,
            iou_thres=args.iou_thres,
            score_thres=args.conf_thres,
            device=device,
            ort=args.ort,
            trt_version=args.trt_version,
            with_preprocess=args.with_preprocess,
        )

    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to("cuda")
    _ = model(dummy_input)
    torch.onnx._export(
        model,
        dummy_input,
        onnx_name,
        input_names=[args.input],
        output_names=["num_dets", "det_boxes", "det_scores", "det_classes"]
        if args.end2end
        else [args.output],
        opset_version=args.opset,
    )
    onnx_model = onnx.load(onnx_name)
    # Fix output shape
    if args.end2end and not args.ort:
        shapes = [
            args.batch_size,
            1,
            args.batch_size,
            args.topk_all,
            4,
            args.batch_size,
            args.topk_all,
            args.batch_size,
            args.topk_all,
        ]
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))

    try:
        import onnxsim

        logger.info("Starting to simplify ONNX...")
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, "check failed"
    except Exception as e:
        logger.info(f"simplify failed: {e}")
    onnx.save(onnx_model, onnx_name)
    logger.info("generated onnx model named {}".format(onnx_name))
    if args.trt:
        trt_name = trt_export(onnx_name, args.batch_size, args.img_size, args.img_size, args.half)
        if args.trt_eval:
            from trt_eval import trt_inference

            logger.info("start trt inference on coco validataion dataset")
            trt_inference(
                config,
                trt_name,
                args.img_size,
                args.batch_size,
                args.conf_thres,
                args.iou_thres,
                args.end2end,
            )


if __name__ == "__main__":
    main()
