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

#### 1.2 SAM2

> 这是 SAM2

好像之前是在 Linux 上运行的
