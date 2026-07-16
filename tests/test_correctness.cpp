#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include "src/models/llama/llama.h"
#include "src/models/llama/llama_params.h"
#include "src/kernels/cublas_utils.h"
#include "src/memory/allocator/cuda_allocator.h"

// Helper to compute cosine similarity between two vectors
float cosine_similarity(const std::vector<float>& A, const std::vector<float>& B) {
    if (A.size() != B.size() || A.empty()) return 0.0f;
    
    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    
    for (size_t i = 0; i < A.size(); ++i) {
        dot_product += A[i] * B[i];
        norm_a += A[i] * A[i];
        norm_b += B[i] * B[i];
    }
    
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return static_cast<float>(dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <weights_dir> <hf_logits.bin> <input_ids.txt>" << std::endl;
        return 1;
    }
    
    std::string weights_dir = argv[1];
    std::string hf_logits_path = argv[2];
    std::string input_ids_path = argv[3];
    
    // 1. Read input IDs
    std::vector<int> input_ids;
    std::ifstream ids_file(input_ids_path);
    int id;
    while (ids_file >> id) {
        input_ids.push_back(id);
    }
    ids_file.close();
    
    std::cout << "Read " << input_ids.size() << " input IDs." << std::endl;
    
    // 2. Read HF reference logits
    std::vector<float> hf_logits;
    std::ifstream logits_file(hf_logits_path, std::ios::binary | std::ios::ate);
    if (!logits_file) {
        std::cerr << "Failed to open " << hf_logits_path << std::endl;
        return 1;
    }
    std::streamsize size = logits_file.tellg();
    logits_file.seekg(0, std::ios::beg);
    int vocab_size = size / sizeof(float);
    hf_logits.resize(vocab_size);
    if (!logits_file.read(reinterpret_cast<char*>(hf_logits.data()), size)) {
        std::cerr << "Failed to read logits" << std::endl;
        return 1;
    }
    std::cout << "Read HF logits with vocab size: " << vocab_size << std::endl;
    
    // 3. Initialize InferSpore components
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
    
    // Dummy cublasLt handle
    cublasLtHandle_t cublaslt_handle;
    cublasLtCreate(&cublaslt_handle);
    
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    cublas_wrapper->setFP32GemmConfig();
    
    CudaAllocator* allocator = new CudaAllocator();
    
    // 4. Initialize Llama Model
    // We assume a standard LLaMA-7B configuration for this test
    // In a real app, this should be parsed from a config file.
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
    int max_seq_len = 2048;
    
    std::cout << "Creating Llama model..." << std::endl;
    // We use FP32 for the correctness test to avoid precision issues hiding bugs
    Llama<float> model(head_num, kv_head_num, head_size, inter_size, num_layers,
                       vocab_size, attn_params, max_seq_len,
                       stream, cublas_wrapper, allocator);
                       
    std::cout << "Loading weights from " << weights_dir << "..." << std::endl;
    model.loadWeights(weights_dir);
    
    // 5. Run Context Decoder (Prefill)
    std::cout << "Running context decoder..." << std::endl;
    int batch_size = 1;
    int max_q_len = input_ids.size();
    
    model.allocateBuffers(batch_size, max_q_len, max_seq_len);
    
    int h_input_lengths[1] = { (int)input_ids.size() };
    int h_history_lengths[1] = { 0 };
    
    model.contextDecode(input_ids.data(), h_input_lengths, h_history_lengths,
                        batch_size, max_q_len, max_q_len);
                        
    // 6. Extract final logits
    std::vector<float> is_logits(vocab_size);
    // logits_buf_ data pointer is public via TensorWrapper
    cudaMemcpy(is_logits.data(), model.logits_buf_->data, 
               vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
               
    // 7. Compare logits
    float similarity = cosine_similarity(hf_logits, is_logits);
    std::cout << "========================================" << std::endl;
    std::cout << "Cosine Similarity: " << similarity << std::endl;
    
    // Print top 5 from both
    auto get_top5 = [](const std::vector<float>& vec) {
        std::vector<std::pair<float, int>> indexed;
        for (int i = 0; i < vec.size(); ++i) {
            indexed.push_back({vec[i], i});
        }
        std::sort(indexed.rbegin(), indexed.rend());
        std::vector<std::pair<float, int>> top5;
        for (int i = 0; i < 5; ++i) top5.push_back(indexed[i]);
        return top5;
    };
    
    auto hf_top5 = get_top5(hf_logits);
    auto is_top5 = get_top5(is_logits);
    
    std::cout << "\nTop 5 predictions (HF Reference):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << i+1 << ". ID: " << hf_top5[i].second << ", Logit: " << hf_top5[i].first << std::endl;
    }
    
    std::cout << "\nTop 5 predictions (InferSpore):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << i+1 << ". ID: " << is_top5[i].second << ", Logit: " << is_top5[i].first << std::endl;
    }
    
    std::cout << "========================================" << std::endl;
    
    if (similarity > 0.99f) {
        std::cout << "SUCCESS: Model output matches reference implementation!" << std::endl;
    } else {
        std::cout << "FAILURE: Model output deviates significantly from reference." << std::endl;
    }
    
    model.freeBuffers();
    delete allocator;
    delete cublas_wrapper;
    cublasLtDestroy(cublaslt_handle);
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(stream);
    
    return 0;
}
