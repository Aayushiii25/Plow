#pragma once
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "src/models/basemodel.h"
#include "src/models/llama/llama_params.h"
#include "src/models/tokenizer.h"
#include "src/weights/llama/llama_weights.h"
#include "src/utils/tensor.h"
#include "src/memory/allocator/base_allocator.h"
#include "src/kernels/cublas_utils.h"

// Kernel launch declarations
#include "src/kernels/input_embedding.h"
#include "src/kernels/cal_paddingoffset.h"
#include "src/kernels/build_casual_mask.h"
#include "src/kernels/qkv_bias_and_RoPE.h"
#include "src/kernels/concat_past_kv.h"
#include "src/kernels/repeat_kv.h"
#include "src/kernels/attn_softmax_kernel.h"
#include "src/kernels/fused_addresidual_norm.h"
#include "src/kernels/fused_decoder_self_attention.h"
#include "src/kernels/linear.h"
#include "src/kernels/act_kernel.h"
#include "src/kernels/topK.h"

template<typename T>
class Llama : public BaseModel {
private:
    // Model config
    int     head_num_;
    int     kv_head_num_;
    int     head_size_;
    int     hidden_units_;    // head_num * head_size
    int     inter_size_;
    int     num_layers_;
    int     vocab_size_;
    int     max_seq_len_;
    float   rmsnorm_eps_;
    bool    attn_bias_;

    LLaMAAttentionStaticParams attn_static_params_;

    // Weights
    LlamaWeight<T>* weights_ = nullptr;

    // Tokenizer
    Tokenizer tokenizer_;
    int bos_token_id_;
    int eos_token_id_;

    // ========== GPU Buffers ==========
    // --- Shared / reused ---
    TensorWrapper<int>*  input_ids_buf_      = nullptr;  // [token_num] or [batch_size]
    TensorWrapper<T>*    decoder_input_buf_   = nullptr;  // [token_num, hidden_units]
    TensorWrapper<T>*    decoder_output_buf_  = nullptr;  // [token_num, hidden_units]
    TensorWrapper<T>*    decoder_residual_buf_ = nullptr; // [token_num, hidden_units]

    // --- Context decoder (prefill) ---
    TensorWrapper<int>*  padding_offset_      = nullptr;  // [batch_size, max_q_len]
    TensorWrapper<int>*  cum_seqlens_         = nullptr;  // [batch_size + 1]
    TensorWrapper<T>*    qkv_buf_             = nullptr;  // [token_num, qkv_head_num, head_size]
    TensorWrapper<T>*    q_buf_               = nullptr;  // [batch_size, head_num, max_q_len, head_size]
    TensorWrapper<T>*    k_buf_               = nullptr;  // [batch_size, kv_head_num, max_q_len, head_size]
    TensorWrapper<T>*    v_buf_               = nullptr;  // [batch_size, kv_head_num, max_q_len, head_size]
    TensorWrapper<T>*    qk_buf_              = nullptr;  // [batch_size, head_num, max_q_len, max_k_len]
    TensorWrapper<T>*    attn_score_buf_      = nullptr;  // [batch_size, head_num, max_q_len, max_k_len]
    TensorWrapper<T>*    attn_output_buf_     = nullptr;  // [batch_size, head_num, max_q_len, head_size]
    TensorWrapper<T>*    attn_proj_buf_       = nullptr;  // [token_num, hidden_units]
    TensorWrapper<T>*    mask_buf_            = nullptr;  // [batch_size, max_q_len, max_k_len]
    TensorWrapper<T>*    k_cache_repeated_    = nullptr;  // [batch_size, head_num, max_k_len, head_size]
    TensorWrapper<T>*    v_cache_repeated_    = nullptr;  // [batch_size, head_num, max_k_len, head_size]
    TensorWrapper<T>*    ctx_attn_output_     = nullptr;  // [token_num, hidden_units]

    // --- Self decoder (token gen) ---
    TensorWrapper<T>*    self_qkv_buf_        = nullptr;  // [batch_size, qkv_head_num, head_size]
    TensorWrapper<T>*    self_mha_output_     = nullptr;  // [batch_size, hidden_units]

    // --- FFN ---
    TensorWrapper<T>*    gate_up_buf_         = nullptr;  // [token_num, 2, inter_size]
    TensorWrapper<T>*    ffn_inter_buf_       = nullptr;  // [token_num, inter_size]
    TensorWrapper<T>*    ffn_output_buf_      = nullptr;  // [token_num, hidden_units]

    // --- KV Cache ---
    TensorWrapper<T>*    all_k_cache_         = nullptr;  // [num_layers, batch_size, kv_head_num, max_seq_len, head_size]
    TensorWrapper<T>*    all_v_cache_         = nullptr;  // [num_layers, batch_size, kv_head_num, max_seq_len, head_size]

    // --- LM Head ---
    TensorWrapper<T>*    logits_buf_          = nullptr;  // [batch_size, vocab_size]

    // --- Sampling ---
    TensorWrapper<int>*  topk_ids_buf_        = nullptr;
    TensorWrapper<T>*    topk_vals_buf_       = nullptr;
    TensorWrapper<int>*  final_topk_ids_      = nullptr;
    TensorWrapper<T>*    final_topk_vals_     = nullptr;
    TensorWrapper<int>*  output_token_ids_    = nullptr;  // [batch_size]

    // --- Metadata ---
    TensorWrapper<int>*  input_lengths_       = nullptr;  // [batch_size]
    TensorWrapper<int>*  history_lengths_     = nullptr;  // [batch_size]
    TensorWrapper<int>*  context_lengths_     = nullptr;  // [batch_size]
    TensorWrapper<int>*  layer_id_            = nullptr;  // scalar on CPU
    TensorWrapper<int>*  step_                = nullptr;  // scalar on CPU
    TensorWrapper<bool>* finished_            = nullptr;  // [batch_size]

    // --- Device pointers (raw allocations) ---
    T*   d_decoder_input_   = nullptr;
    T*   d_decoder_output_  = nullptr;
    T*   d_decoder_residual_= nullptr;
    int* d_input_ids_       = nullptr;
    T*   d_qkv_buf_         = nullptr;
    T*   d_q_buf_           = nullptr;
    T*   d_k_buf_           = nullptr;
    T*   d_v_buf_           = nullptr;
    T*   d_qk_buf_          = nullptr;
    T*   d_attn_score_      = nullptr;
    T*   d_attn_output_     = nullptr;
    T*   d_attn_proj_       = nullptr;
    T*   d_mask_            = nullptr;
    T*   d_k_cache_rep_     = nullptr;
    T*   d_v_cache_rep_     = nullptr;
    T*   d_ctx_attn_out_    = nullptr;
    T*   d_self_qkv_        = nullptr;
    T*   d_self_mha_out_    = nullptr;
    T*   d_gate_up_         = nullptr;
    T*   d_ffn_inter_       = nullptr;
    T*   d_ffn_output_      = nullptr;
    T*   d_all_k_cache_     = nullptr;
    T*   d_all_v_cache_     = nullptr;
    T*   d_logits_          = nullptr;
    int* d_padding_offset_  = nullptr;
    int* d_cum_seqlens_     = nullptr;
    int* d_topk_ids_        = nullptr;
    T*   d_topk_vals_       = nullptr;
    int* d_final_topk_ids_  = nullptr;
    T*   d_final_topk_vals_ = nullptr;
    int* d_output_ids_      = nullptr;
    int* d_input_lengths_   = nullptr;
    int* d_history_lengths_ = nullptr;
    int* d_context_lengths_ = nullptr;
    bool* d_finished_       = nullptr;

    // CPU-side metadata
    int* h_layer_id_        = nullptr;
    int* h_step_            = nullptr;

    // Internal methods
    void allocateBuffers(int batch_size, int max_q_len, int max_k_len);
    void freeBuffers();

    // Context decoder: full prefill
    void contextDecode(int* h_input_ids, int* h_input_lengths, int* h_history_lengths,
                       int batch_size, int max_q_len, int max_k_len);

    // Self decoder: single token generation step
    void selfDecode(int* h_input_ids, int* h_history_lengths,
                    int batch_size, int step);

public:
    Llama(int head_num,
          int kv_head_num,
          int head_size,
          int inter_size,
          int num_layers,
          int vocab_size,
          LLaMAAttentionStaticParams attn_static_params,
          int max_seq_len,
          cudaStream_t stream,
          cublasWrapper* cublas_wrapper,
          BaseAllocator* allocator,
          cudaDeviceProp* cuda_device_prop = nullptr);

    ~Llama();

    void loadTokenizer(std::string file) override;
    void loadWeights(std::string file) override;
    void loadWeightsFromDummy() override;

    std::vector<std::string> MakeInput(const std::string &history, int round, const std::string &input) override;
    std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) override;
    std::string Response(const std::vector<std::string>& input, CallBack PrintRes) override;
};
