#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "src/models/llama/llama.h"
#include "src/models/llama/llama_params.h"
#include "src/kernels/cublas_utils.h"
#include "src/memory/allocator/cuda_allocator.h"

// Print callback for streaming tokens
void print_token(int index, const char* token) {
    std::cout << token << std::flush;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <weights_dir> <tokenizer_path> <prompt>" << std::endl;
        return 1;
    }
    
    std::string weights_dir = argv[1];
    std::string tokenizer_path = argv[2];
    std::string prompt = argv[3];
    
    // Initialize CUDA and cuBLAS
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
    
    cublasLtHandle_t cublaslt_handle;
    cublasLtCreate(&cublaslt_handle);
    
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    cublas_wrapper->setFP32GemmConfig(); // Using FP32 for now
    
    CudaAllocator* allocator = new CudaAllocator();
    
    // Model configuration (LLaMA-7B defaults)
    LLaMAAttentionStaticParams attn_params;
    attn_params.rotary_embedding_dim = 128;
    attn_params.rotary_embedding_base = 10000.0f;
    attn_params.max_position_embeddings = 2048;
    attn_params.use_dynamic_ntk = false;
    
    int head_num = 32;
    int kv_head_num = 32;
    int head_size = 128;
    int inter_size = 11008;
    int num_layers = 32;
    int vocab_size = 32000;
    int max_seq_len = 2048;
    
    // std::cerr << "[InferSpore] Loading model..." << std::endl;
    // std::cerr stream is used for logs so it doesn't pollute stdout (which the web server reads)
    
    Llama<float> model(head_num, kv_head_num, head_size, inter_size, num_layers,
                       vocab_size, attn_params, max_seq_len,
                       stream, cublas_wrapper, allocator);
                       
    model.loadTokenizer(tokenizer_path);
    model.loadWeights(weights_dir);
    
    // std::cerr << "[InferSpore] Generation started" << std::endl;
    
    std::vector<std::string> input;
    input.push_back(prompt);
    
    // Generate response. Tokens are printed to stdout via the print_token callback.
    model.Response(input, print_token);
    
    std::cout << std::endl;
    
    delete allocator;
    delete cublas_wrapper;
    cublasLtDestroy(cublaslt_handle);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    
    return 0;
}
