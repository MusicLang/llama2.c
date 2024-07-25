!git clone https://github.com/ggerganov/llama.cpp
!apt install nvidia-cuda-toolkit
!cd llama.cpp && make GGML_CUDA=1
from huggingface_hub import snapshot_download
model_name = "musiclang/control_masking_optimized_trained"
methods = ['q4_k_m']
base_model = "./original_model/"
quantized_path = "./quantized_model/"
# prompt: Login to huggingface

from huggingface_hub import notebook_login
notebook_login()
snapshot_download(repo_id=model_name, local_dir=base_model , local_dir_use_symlinks=False)
original_model = quantized_path+'/FP16.gguf'
!mkdir ./quantized_model/
!python llama.cpp/convert_hf_to_gguf.py original_model --outtype f16 --outfile ./quantized_model/FP16.gguf
!python llama.cpp/convert_hf_to_gguf.py original_model --outtype f32 --outfile ./quantized_model/FP32.gguf