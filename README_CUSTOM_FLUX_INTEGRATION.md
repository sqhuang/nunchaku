# 自定义 Flux 模型 PTQ 量化集成指南

本指南详细介绍如何将基于 `nn.Module` 的自定义 Flux 模型集成到 DeepCompressor/Nunchaku 框架中进行 SVDQuant PTQ 量化。

## 目录

1. [概述](#概述)
2. [环境准备](#环境准备)
3. [集成步骤](#集成步骤)
4. [详细实现](#详细实现)
5. [使用示例](#使用示例)
6. [常见问题](#常见问题)
7. [性能优化](#性能优化)

## 概述

### 什么是 SVDQuant？

SVDQuant 是一种针对扩散模型的 4-bit 权重和激活量化技术，通过以下方式实现：

1. **激活值迁移**：将激活值中的异常值迁移到权重中
2. **SVD 分解**：使用奇异值分解将难以量化的权重分解为低秩组件
3. **混合精度**：低秩组件保持 16-bit，残差部分进行 4-bit 量化

### 架构概览

```
自定义 Flux 模型 (nn.Module)
    ↓
包装器 (FluxTransformer2DModel 兼容)
    ↓
DeepCompressor 量化 (SVDQuant)
    ↓
Nunchaku 后端 (C++ 优化推理)
    ↓
量化推理 (3.6x 内存减少, 8.7x 速度提升)
```

## 环境准备

### 1. 安装依赖

```bash
# 基础依赖
conda create -n flux_quantization python=3.11
conda activate flux_quantization

# PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Diffusers
pip install diffusers transformers accelerate

# DeepCompressor (从源码安装)
git clone https://github.com/mit-han-lab/deepcompressor.git
cd deepcompressor
pip install poetry
poetry install

# Nunchaku (从源码安装)
git clone https://github.com/mit-han-lab/nunchaku.git
cd nunchaku
git submodule init
git submodule update
pip install -e .
```

### 2. 验证安装

```python
# 运行提供的验证脚本
python deepcompressor_integration.py
```

## 集成步骤

### 步骤 1: 包装你的自定义模型

创建一个继承自 `FluxTransformer2DModel` 的包装器：

```python
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import register_to_config

class CustomFluxWrapper(FluxTransformer2DModel):
    @register_to_config
    def __init__(self, custom_model_path=None, **kwargs):
        super().__init__(num_layers=0, num_single_layers=0, **kwargs)
        
        # 加载你的自定义模型
        self.custom_model = self._load_custom_model(custom_model_path)
        
        # 清除标准层
        self.transformer_blocks = nn.ModuleList([])
        self.single_transformer_blocks = nn.ModuleList([])
    
    def _load_custom_model(self, model_path):
        # 加载你的自定义模型实现
        return YourCustomFluxModel.from_pretrained(model_path)
    
    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        # 调用你的自定义模型
        return self.custom_model(hidden_states, encoder_hidden_states, **kwargs)
```

### 步骤 2: 准备量化数据

```python
def create_calibration_dataset():
    """创建校准数据集"""
    calibration_data = []
    
    for i in range(512):  # 推荐 512 个样本
        sample = {
            'hidden_states': torch.randn(1, 4096, 3072),
            'encoder_hidden_states': torch.randn(1, 512, 4096),
            'timestep': torch.randint(0, 1000, (1,)),
            # 根据你的模型添加其他输入
        }
        calibration_data.append(sample)
    
    return calibration_data
```

### 步骤 3: 运行量化

```python
from deepcompressor_integration import complete_quantization_pipeline

# 加载你的自定义模型
custom_model = YourCustomFluxModel()

# 运行完整的量化流水线
quantized_model = complete_quantization_pipeline(
    custom_flux_model=custom_model,
    output_path="./quantized_custom_flux"
)
```

### 步骤 4: 集成到推理管道

```python
from diffusers import FluxPipeline

# 创建推理管道
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    transformer=quantized_model,
    torch_dtype=torch.bfloat16
).to("cuda")

# 生成图像
image = pipeline(
    "A beautiful landscape",
    num_inference_steps=4,
    guidance_scale=0
).images[0]
```

## 详细实现

### 自定义模型要求

你的自定义 Flux 模型需要满足以下接口：

```python
class YourCustomFluxModel(nn.Module):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        **kwargs
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim] 图像潜在表示
            encoder_hidden_states: [batch, text_len, text_dim] 文本编码
            timestep: [batch] 时间步
            
        Returns:
            torch.FloatTensor: 预测的噪声
        """
        # 你的模型实现
        return predicted_noise
```

### 关键配置参数

```python
@dataclass
class QuantizationConfig:
    weight_bits: int = 4        # 权重量化位数
    activation_bits: int = 4    # 激活量化位数
    group_size: int = 128       # 量化组大小
    svd_rank: int = 32          # SVD 低秩分解的秩
    calibration_samples: int = 512  # 校准样本数量
```

### 性能调优参数

```python
# 根据你的 GPU 内存调整批处理大小
calibration_config = {
    "batch_size": 1,            # 校准批大小
    "num_samples": 512,         # 校准样本总数
    "device": "cuda",           # 设备
}

# SVD 配置
svd_config = {
    "rank": 32,                 # 较大的秩 = 更好的质量，但更多内存
    "enable": True,             # 启用 SVD 分解
}
```

## 使用示例

### 基础示例

```python
import torch
from custom_flux_integration import CustomFluxWrapper
from deepcompressor_integration import complete_quantization_pipeline

# 1. 加载你的模型
model = CustomFluxWrapper(
    custom_model_path="/path/to/your/flux/model.pth"
)

# 2. 量化
quantized_model = complete_quantization_pipeline(
    custom_flux_model=model,
    output_path="./my_quantized_flux"
)

# 3. 推理
input_data = {
    'hidden_states': torch.randn(1, 4096, 3072).cuda(),
    'encoder_hidden_states': torch.randn(1, 512, 4096).cuda(),
    'timestep': torch.tensor([500]).cuda(),
}

with torch.no_grad():
    output = quantized_model(**input_data)
    print(f"输出形状: {output.shape}")
```

### ComfyUI 集成示例

```python
class CustomFluxComfyUINode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "/path/to/quantized/model"}),
                "device_id": ("INT", {"default": 0}),
            }
        }

    def load_model(self, model_path: str, device_id: int):
        from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel
        
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            model_path
        ).to(f"cuda:{device_id}")
        
        return (transformer,)
```

## 常见问题

### Q1: 内存不足怎么办？

**A:** 尝试以下解决方案：

```python
# 减少校准样本数量
config.calibration_samples = 256

# 减少 SVD 秩
config.svd_rank = 16

# 使用梯度检查点
torch.autograd.set_grad_enabled(False)

# 清理 GPU 缓存
torch.cuda.empty_cache()
```

### Q2: 量化质量不好怎么办？

**A:** 调整以下参数：

```python
# 增加校准样本
config.calibration_samples = 1024

# 使用更大的 SVD 秩
config.svd_rank = 64

# 使用更高质量的校准数据
# 确保校准数据覆盖模型的典型使用场景
```

### Q3: 推理速度没有提升？

**A:** 检查以下方面：

```python
# 确保使用 Nunchaku 后端
assert isinstance(model, NunchakuFluxTransformer2dModel)

# 禁用梯度计算
with torch.no_grad():
    output = model(input)

# 预热 GPU
for _ in range(5):
    _ = model(warmup_input)
```

### Q4: 如何验证量化效果？

```python
def compare_models(original_model, quantized_model, test_inputs):
    """比较原始模型和量化模型的输出"""
    
    with torch.no_grad():
        original_output = original_model(**test_inputs)
        quantized_output = quantized_model(**test_inputs)
        
        # 计算差异
        mse = torch.nn.functional.mse_loss(original_output, quantized_output)
        cosine_sim = torch.nn.functional.cosine_similarity(
            original_output.flatten(), 
            quantized_output.flatten(), 
            dim=0
        )
        
        print(f"MSE: {mse.item():.6f}")
        print(f"Cosine Similarity: {cosine_sim.item():.6f}")
```

## 性能优化

### 内存优化

```python
# 1. 使用混合精度
model = model.half()  # 转换为 fp16

# 2. 梯度累积
for i, batch in enumerate(calibration_data):
    if i % accumulation_steps == 0:
        torch.cuda.empty_cache()

# 3. 分块处理
def process_in_chunks(data, chunk_size=32):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]
```

### 推理优化

```python
# 1. 启用 CUDA 图
torch.backends.cudnn.benchmark = True

# 2. 使用编译优化
model = torch.compile(model, mode="reduce-overhead")

# 3. 批处理推理
batch_inputs = torch.stack([input1, input2, input3])
batch_outputs = model(batch_inputs)
```

### 模型大小优化

```python
# 选择合适的 SVD 秩
rank_vs_quality = {
    16: "最小模型，质量较低",
    32: "平衡选择，推荐",
    64: "高质量，较大模型"
}

# 动态调整量化参数
if model_size > threshold:
    config.weight_bits = 4
    config.activation_bits = 4
else:
    config.weight_bits = 8
    config.activation_bits = 8
```

## 故障排除

### 常见错误及解决方案

1. **CUDA out of memory**
   ```python
   # 减少批处理大小
   config.batch_size = 1
   # 清理缓存
   torch.cuda.empty_cache()
   ```

2. **Import Error: deepcompressor**
   ```bash
   # 重新安装 DeepCompressor
   pip uninstall deepcompressor
   git clone https://github.com/mit-han-lab/deepcompressor.git
   cd deepcompressor && pip install -e .
   ```

3. **Model shape mismatch**
   ```python
   # 检查模型接口兼容性
   assert hasattr(model, 'forward')
   assert model.forward.__code__.co_varnames[:3] == ('self', 'hidden_states', 'encoder_hidden_states')
   ```

## 高级特性

### 自定义量化策略

```python
class CustomQuantizationStrategy:
    def __init__(self):
        self.layer_configs = {}
    
    def configure_layer(self, layer_name, bits=4, group_size=128):
        self.layer_configs[layer_name] = {
            'bits': bits,
            'group_size': group_size
        }
    
    def apply(self, model):
        for name, module in model.named_modules():
            if name in self.layer_configs:
                config = self.layer_configs[name]
                # 应用自定义配置
```

### LoRA 集成

```python
def add_lora_support(quantized_model, lora_path):
    """为量化模型添加 LoRA 支持"""
    quantized_model.update_lora_params(lora_path)
    quantized_model.set_lora_strength(1.0)
    return quantized_model
```

这个指南提供了将自定义 Flux 模型集成到 DeepCompressor 进行 PTQ 量化的完整流程。按照这些步骤，你应该能够成功量化你的模型并获得显著的性能提升。