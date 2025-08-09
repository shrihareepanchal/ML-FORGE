# Llama 3.2 Setup Guide for ML-FORGE

This guide will help you set up Llama 3.2 for the ML-FORGE Chat Assistant feature. The chat assistant allows you to ask questions about machine learning concepts and receive informative responses from a locally running LLM.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning repositories)
- C++ compiler (for Windows: Visual Studio Build Tools)

## Step 1: Install Required Dependencies

### Windows

1. Install Visual Studio Build Tools:
   - Download from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
   - Select "Visual Studio Build Tools"
   - During installation, select "Desktop development with C++"
   - Install the package

2. Install the llama-cpp-python package:

   **CPU-only version:**
   ```bash
   pip install llama-cpp-python
   ```

   **GPU-accelerated version (recommended if you have a compatible NVIDIA GPU):**
   ```bash
   pip install llama-cpp-python --prefer-binary --extra-index-url=https://pypi.anaconda.org/scipy-wheels-nightly/simple
   ```

### macOS

1. Install Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```

2. Install the llama-cpp-python package:
   ```bash
   pip install llama-cpp-python
   ```

   For Metal GPU acceleration:
   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
   ```

### Linux

1. Install build tools:
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential
   ```

2. Install the llama-cpp-python package:
   ```bash
   pip install llama-cpp-python
   ```

   For CUDA support:
   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
   ```

## Step 2: Download a Llama 3.2 Model

1. Visit [TheBloke's Hugging Face page](https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/tree/main)
2. Download a GGUF format model file. We recommend:
   - `llama-3.2-8b-instruct.Q4_K_M.gguf` - Good balance of quality and speed
   - `llama-3.2-8b-instruct.Q2_K.gguf` - Faster but lower quality
   - `llama-3.2-8b-instruct.Q6_K.gguf` - Higher quality but slower

3. Save the model file to a location on your computer, e.g., `C:\llama` on Windows or `~/llama` on macOS/Linux

## Step 3: Configure the Chat Assistant in ML-XPERT

1. Run ML-FORGE:
   ```bash
   streamlit run app.py
   ```

2. Navigate to the "ML Chat Assistant" section in the sidebar

3. In the LLM Model Settings panel:
   - Enter the path to your model file (e.g., `C:/llama/llama-3.2-8b-instruct.Q4_K_M.gguf`)
   - Adjust settings based on your hardware:
     - **n_gpu_layers**: Set to 0 for CPU-only, or a higher value (20-40) for GPU
     - **Context length**: Affects the amount of conversation history remembered
     - **Batch size**: Higher values can be faster but require more memory
     - **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = more creative)
     - **Max tokens**: Maximum length of generated responses

4. Click "Load Model" to initialize the Llama model

## Troubleshooting

### Common Issues

1. **Model Fails to Load**:
   - Ensure the model path is correct
   - Check if you have enough RAM/VRAM
   - Try reducing n_gpu_layers or setting it to 0
   - Verify that the model file is not corrupted (re-download if necessary)

2. **Slow Responses**:
   - Use a quantized model (Q4_K_M or Q2_K)
   - Reduce context length and batch size
   - If using CPU, consider switching to GPU if available

3. **"CUDA out of memory" Error**:
   - Reduce n_gpu_layers
   - Use a smaller model or more highly quantized version (e.g., Q2_K instead of Q6_K)
   - Close other applications using GPU memory

4. **Installation Errors on Windows**:
   - Ensure Visual Studio Build Tools are properly installed
   - Try the pre-built binary: `pip install llama-cpp-python --prefer-binary`

### Advanced Configuration

For advanced users, you can customize the model loading parameters directly in `app.py`:

```python
# Modify these values in the llm_chat_section function
model_path = "path/to/your/model.gguf"
n_gpu_layers = 20  # Adjust based on your GPU
n_ctx = 2048       # Context length
n_batch = 256      # Batch size
```

## Additional Resources

- [Official llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [TheBloke's Hugging Face Models](https://huggingface.co/TheBloke)
- [Llama-cpp-python Documentation](https://github.com/abetlen/llama-cpp-python)
