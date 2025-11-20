# Running on TPU Google VM

The current implementation is optimized for GPU. To run on TPU, follow these modifications:

## Prerequisites

```bash
# Install PyTorch XLA
pip install torch~=2.0.0 torch_xla[tpu]~=2.0.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# Install other requirements
pip install -r requirements.txt
```

## Required Code Modifications

### 1. LSTM Models (bilstm.py, cnn_lstm.py, attention_bilstm.py)

**In each LSTM model file, replace the imports:**

```python
# Add these imports at the top
import torch_xla
import torch_xla.core.xla_model as xm
```

**Replace `get_device()` calls:**

```python
# OLD:
device = get_device()

# NEW:
device = xm.xla_device()
print(f"Using TPU device: {device}")
```

**Replace optimizer steps in training loop:**

```python
# OLD:
loss.backward()
optimizer.step()

# NEW:
loss.backward()
xm.optimizer_step(optimizer)
xm.mark_step()  # Synchronize XLA operations
```

**Replace model saving:**

```python
# OLD:
torch.save(checkpoint, filepath)

# NEW:
xm.save(checkpoint, filepath)
```

### 2. Transformer Models (xlmr.py, banglabert.py)

**Change Accelerator initialization:**

```python
# OLD:
accelerator = Accelerator(mixed_precision='fp16')

# NEW (TPU uses bfloat16):
accelerator = Accelerator(mixed_precision='bf16')
```

### 3. Update utils/training_utils.py

**Replace `get_device()` function:**

```python
def get_device():
    """
    Get available device (TPU if available, else GPU, else CPU)
    """
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"Using TPU: {device}")
        return device
    except ImportError:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device
```

## Performance Considerations

### TPU vs GPU Differences:

1. **Batch Size**: TPUs work best with larger batch sizes (32-128)
   - Increase LSTM batch_size from 32 to 64 or 128
   - Increase Transformer batch_size from 16 to 32 or 64

2. **Data Loading**: Use more workers for DataLoader
   ```python
   train_loader = DataLoader(..., num_workers=8, prefetch_factor=2)
   ```

3. **Mixed Precision**: TPU uses bfloat16 (better range than fp16)

4. **Compilation**: First iteration is slow (XLA compilation)

## Recommended: Use PyTorch/XLA or JAX Instead

For **optimal TPU performance**, consider rewriting with:
- **PyTorch/XLA**: Native TPU support
- **JAX + Flax**: Google's preferred TPU framework
- **TensorFlow**: Also well-optimized for TPU

## Quick Test on TPU

```python
# Test TPU availability
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f"TPU Device: {device}")

# Test basic operation
x = torch.randn(5, 3).to(device)
y = x + 2
print(f"Tensor on TPU: {y}")
```

## Alternative: Run on GPU Instead

If modifications are too complex, use Google Cloud GPU VMs instead:
- **NVIDIA T4**: Good for inference and small models
- **NVIDIA V100**: Better for training
- **NVIDIA A100**: Best performance

The current code runs perfectly on GPU VMs without any changes.
