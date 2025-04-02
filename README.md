# Marine-SAM
### 一、运行方法

#### 1.1 SAM_ORI

```bash
# 环境配置方法
conda create -n SAM_ORI python==3.11 -y
conda activate SAM_ORI
cd D:\BISTU\Graduation_Project\Marine-SAM\SAM_ORI

pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx -i https://mirrors.aliyun.com/pypi/simple
pip3 install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 -i https://download.pytorch.org/whl/cu121
```

单张图片执行：直接运行 `python main.py` 即可，记得更改代码中的图片路径

> 这是 SAM 的原始版本，未经过任何微调，效果还行（但肯定离预期还有一定距离）

![image](./img/SAM_ORI_1.png)

![image](./img/SAM_ORI_2.png)

#### 1.2 SAM2_FT

> 这是 SAM2 的微调版本，目前还没有进行微调，详见：
>
> https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code?tab=readme-ov-file
>
> https://medium.com/data-science/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3

```bash
# Windows 暂未测试，推荐用 WSL
```

```bash
# Linux 实测可正常配置环境：
pip install numpy
pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

注意：需要先 `Train.py` 生成 `model.torch`，再 `TEST_Net.py`，另外，训练代码需要针对 RUOD 数据集进行格式上的调整。
