[English](README.md) | 简体中文

<div align="center"><img src="assets/logo.png" width="1500"></div>

## 简介

欢迎来到**DAMO-YOLO**！DAMO-YOLO 是由阿里巴巴达摩院智能计算实验室 TinyML 团队开发的一个兼顾速度与精度的目标检测框架,其效果超越了目前的一众 YOLO 系列方法，在实现 SOTA 的同时，保持了很高的推理速度。DAMO-YOLO 是在 YOLO 框架基础上引入了一系列新技术，对整个检测框架进行了大幅的修改。具体包括：基于 NAS 搜索的高效检测骨干网络，更深的多尺度特征融合检测颈部，精简的检测头结构，以及引入蒸馏技术实现效果的进一步提升。具体细节可以参考我们的[技术报告](https://arxiv.org/pdf/2211.15444v2.pdf)。模型之外，DAMO-YOLO 还提供高效的训练策略以及便捷易用的部署工具，帮助您快速解决工业落地中的实际问题！

<div align="center"><img src="assets/curve.png" width="500"></div>

## 更新日志

- **[2022/11/27: DAMO-YOLO v0.1.1 更新!]**
  - 增加详细的[自有数据微调模型教程](./assets/CustomDatasetTutorial.md)。
  - 修复了空标签数据导致训练卡住的问题[issue #30](https://github.com/tinyvision/DAMO-YOLO/issues/30)，如您使用中遇到任何问题，欢迎随时反馈，我们 24 小时待命。
- **[2022/11/27: DAMO-YOLO v0.1.0 开源!]**
  - 开源 DAMO-YOLO-T, DAMO-YOLO-S 和 DAMO-YOLO-M 模型。
  - 开源模型导出工具，支持 onnx 导出以及 TensorRT-fp32、TensorRT-fp16 模型导出。

## 线上 Demo

- 线上 Demo 已整合至 ModelScope，快去[DAMO-YOLO-T](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo-t/summary)，[DAMO-YOLO-S](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo/summary)，[DAMO-YOLO-M](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo-m/summary) 体验一下吧！

## 模型库

| Model                                               | size | mAP<sup>val<br>0.5:0.95 | Latency T4<br>TRT-FP16-BS1 | FLOPs<br>(G) | Params<br>(M) |                                                                                                                    Download                                                                                                                     |
| --------------------------------------------------- | :--: | :---------------------: | :------------------------: | :----------: | :-----------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [DAMO-YOLO-T](./configs/damoyolo_tinynasL20_T.py)   | 640  |          41.8           |            2.78            |     18.1     |      8.5      | [torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL20_T_418.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/before_distill/damoyolo_tinynasL20_T_418.onnx) |
| [DAMO-YOLO-T\*](./configs/damoyolo_tinynasL20_T.py) | 640  |          43.0           |            2.78            |     18.1     |      8.5      |                    [torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/damoyolo_tinynasL20_T.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/damoyolo_tinynasL20_T.onnx)                    |
| [DAMO-YOLO-S](./configs/damoyolo_tinynasL25_S.py)   | 640  |          45.6           |            3.83            |     37.8     |     16.3      | [torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL25_S_456.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/before_distill/damoyolo_tinynasL25_S_456.onnx) |
| [DAMO-YOLO-S\*](./configs/damoyolo_tinynasL25_S.py) | 640  |          46.8           |            3.83            |     37.8     |     16.3      |                    [torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/damoyolo_tinynasL25_S.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/damoyolo_tinynasL25_S.onnx)                    |
| [DAMO-YOLO-M](./configs/damoyolo_tinynasL35_M.py)   | 640  |          48.7           |            5.62            |     61.8     |     28.2      | [torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL35_M_487.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/before_distill/damoyolo_tinynasL35_M_487.onnx) |
| [DAMO-YOLO-M\*](./configs/damoyolo_tinynasL35_M.py) | 640  |          50.0           |            5.62            |     61.8     |     28.2      |                    [torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/damoyolo_tinynasL35_M.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/damoyolo_tinynasL35_M.onnx)                    |

- 上表中汇报的是 COCO2017 val 集上的结果, 测试时使用 multi-class NMS。
- 其中 latency 中不包括后处理时间。
- \* 表示模型训练时使用蒸馏。

## 快速上手

<details>
<summary>安装</summary>

步骤一. 安装 DAMO-YOLO.

```shell
git clone https://github.com/tinyvision/DAMO-YOLO.git
cd DAMO-YOLO/
conda create -n DAMO-YOLO python=3.7 -y
conda activate DAMO-YOLO
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

步骤二. 安装[pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython;
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI # for Linux
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI # for Windows
```

</details>

<details>
<summary>Demo</summary>

步骤一. 从模型库中下载训练好的 torch 模型或 onnx 推理引擎，例如 damoyolo_tinynasL25_S.pth 或 damoyolo_tinynasL25_S.onnx

步骤二. 执行命令时用-f 选项指定配置(config)文件。例如:

```shell
# torch 推理
python tools/torch_inference.py -f configs/damoyolo_tinynasL25_S.py --ckpt /path/to/your/damoyolo_tinynasL25_S.pth --path assets/dog.jpg

# onnx 推理
python tools/onnx_inference.py -f configs/damoyolo_tinynasL25_S.py --onnx /path/to/your/damoyolo_tinynasL25_S.onnx --path assets/dog.jpg
```

</details>

<details>
<summary>从头开始，复现COCO上的精度</summary>

步骤一. 准备好 COCO 数据集,推荐将 coco 数据软链接到 datasets 目录下。

```shell
cd <DAMO-YOLO Home>
ln -s /path/to/your/coco ./datasets/coco
```

步骤二. 在 COCO 数据上进行训练，使用-f 选项指定配置(config)文件。

```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/damoyolo_tinynasL25_S.py
```

</details>

<details>
<summary>在自定义数据上微调模型</summary>

请参考[自有数据微调模型教程](./assets/CustomDatasetTutorial.md)。

</details>

<details>
<summary>在COCO val上测评训练好的模型</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/eval.py -f configs/damoyolo_tinynasL25_S.py --ckpt /path/to/your/damoyolo_tinynasL25_S.pth
```

</details>

<details>
<summary>使用TinyNAS自定义DAMO-YOLO骨干网络</summary>

步骤 1. 如果您想自定义 DAMO-YOLO 骨干网络，可以参考[适用于 DAMO-YOLO 的 MAE-NAS 教程](https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/scripts/damo-yolo/Tutorial_NAS_for_DAMO-YOLO_cn.md)，通过该教程您可以一步步学习如何使用 latency/flops 作为约束条件搜索该条件下的最优模型。

步骤 2. 模型搜索结束后，您可以使用搜索得到的模型结构文件替换 config 中的 structure text。把 Backbone 的 name 设置成 TinyNAS_res 或者 TinyNAS_csp，将会分别得到 ResNet 或者 CSPNet 形式的 TinyNAS 骨干网络, 请注意到 TinyNAS_res 骨干网络的 out_indices=(2,4,5)而 TinyNAS_csp 骨干网络的 out_indices=(2,3,4)。

```
structure = self.read_structure('tinynas_customize.txt')
TinyNAS = { 'name'='TinyNAS_res', # ResNet形式的Tinynas骨干网络
            'out_indices': (2,4,5)}
TinyNAS = { 'name'='TinyNAS_csp', # CSPNet形式的Tinynas骨干网络
            'out_indices': (2,3,4)}

```

</details>

## 部署

<details>
<summary>安装依赖项</summary>

步骤 1. 安装 ONNX.

```shell
pip install onnx==1.8.1
pip install onnxruntime==1.8.0
pip install onnx-simplifier==0.3.5
```

步骤 2. 安装 CUDA、CuDNN、TensorRT and pyCUDA

2.1 CUDA

```shell
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
source ~/.bashrc
```

2.2 CuDNN

```shell
sudo cp cuda/include/* /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

2.3 TensorRT

```shell
cd TensorRT-7.2.1.6/python
pip install tensorrt-7.2.1.6-cp37-none-linux_x86_64.whl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-7.2.1.6/lib
```

2.4 pycuda

```shell
pip install pycuda==2022.1
```

</details>

<details>
<summary>模型导出</summary>

步骤一：将 torch 模型导出成 onnx 或者 TensorRT 推理引擎。具体使用方法如下：

```shell
# onnx 导出
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640

# trt 导出
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640 --trt --end2end --trt_eval
```

其中--end2end 表示在导出的 onnx 或者 TensorRT 引擎中集成 NMS 模块，--trt_eval 表示在 TensorRT 导出完成后即在 coco2017 val 上进行精度验证。

已经完成 TensorRT 导出的模型也可由如下指令在 coco2017 val 上进行精度验证。--end2end 表示待测试的 TensorRT 引擎包含 NMS 组件。

```shell
python tools/trt_eval.py -f configs/damoyolo_tinynasL25_S.py -trt deploy/damoyolo_tinynasL25_S_end2end.trt --batch_size 1 --img_size 640 --end2end
```

步骤二：使用已经导出的 onnx 或 TensorRT 引擎进行目标检测。

```shell
# onnx 推理
python tools/onnx_inference.py -f configs/damoyolo_tinynasL25_S.py --onnx /path/to/your/damoyolo_tinynasL25_S.onnx --path assets/dog.jpg

# trt 推理
python tools/trt_inference.py -f configs/damoyolo_tinynasL25_s.py -t deploy/damoyolo_tinynasL25_S_end2end_fp16_bs1.trt -p assets/dog.jpg --img_size 640 --end2end
```

</details>

## 实习生招聘

我们正在招聘研究型实习生，如果您对目标检测/模型量化/神经网络搜索等方向有兴趣，敬请将简历投递到xiuyu.sxy@alibaba-inc.com。

## 引用

```latex
 @article{damoyolo,
   title={DAMO-YOLO: A Report on Real-Time Object Detection Design},
   author={Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang and Xiuyu Sun},
   journal={arXiv preprint arXiv:2211.15444v2},
   year={2022},
 }
```
