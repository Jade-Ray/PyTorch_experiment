# The fused window process for acceleration package of [SwinTransformer]('https://github.com/microsoft/Swin-Transformer/tree/afeb877fba1139dfbc186276983af2abb02c2196/kernels')

* Install CUDA>=10.2 with cudnn>=7 following the [official installation instructions]('https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html')

* Install fused window process for acceleration, activated by passing --fused_window_process in the running script

```bash
cd swin_window_process
python setup.py install #--user
```
