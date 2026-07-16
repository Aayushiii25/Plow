#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include "src/models/llama/llama.h"
#include "src/utils/macro.h"

// ========================== Constructor ==========================
template<typename T>
Llama<T>::Llama(int head_num,
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
                cudaDeviceProp* cuda_device_prop)
    : BaseModel(stream, cublas_wrapper, allocator, cuda_device_prop),
      head_num_(head_num),
      kv_head_num_(kv_head_num),
      head_size_(head_size),
      hidden_units_(head_num * head_size),
      inter_size_(inter_size),
      num_layers_(num_layers),
      vocab_size_(vocab_size),
      max_seq_len_(max_seq_len),
      rmsnorm_eps_(1e-5f),
      attn_bias_(false),
      attn_static_params_(attn_static_params),
      bos_token_id_(1),
      eos_token_id_(2)
{
    model_name = "llama";
    WeightType wtype = getWeightType<T>();
    weights_ = new LlamaWeight<T>(head_num, kv_head_num, head_size,
                                   inter_size, vocab_size, num_layers,
                                   attn_bias_, wtype);
    h_layer_id_ = new int[1];
    h_step_     = new int[1];
}

// ========================== Destructor ==========================
template<typename T>
Llama<T>::~Llama() {
    freeBuffers();
    delete weights_;
    delete[] h_layer_id_;
    delete[] h_step_;
}

// ========================== Load Tokenizer ==========================
template<typename T>
void Llama<T>::loadTokenizer(std::string file) {
    tokenizer_.Initialize(file);
    std::cout << "[InferSpore] Tokenizer loaded from " << file << std::endl;
}

// ========================== Load Weights ==========================
template<typename T>
void Llama<T>::loadWeights(std::string file) {
    weights_->loadWeights(file);
    std::cout << "[InferSpore] Weights loaded from " << file << std::endl;
}

template<typename T>
void Llama<T>::loadWeightsFromDummy() {
    weights_->loadWeightsFromDummy();
    std::cout << "[InferSpore] Dummy weights loaded" << std::endl;
}

// ========================== Buffer Management ==========================
template<typename T>
void Llama<T>::allocateBuffers(int batch_size, int max_q_len, int max_k_len) {
    int qkv_head_num = head_num_ + 2 * kv_head_num_;
    int token_num = batch_size * max_q_len; // max possible tokens

    // Allocate device memory
    CHECK(cudaMalloc(&d_input_ids_,       sizeof(int) * token_num));
    CHECK(cudaMalloc(&d_decoder_input_,   sizeof(T) * token_num * hidden_units_));
    CHECK(cudaMalloc(&d_decoder_output_,  sizeof(T) * token_num * hidden_units_));
    CHECK(cudaMalloc(&d_decoder_residual_,sizeof(T) * token_num * hidden_units_));
    CHECK(cudaMalloc(&d_padding_offset_,  sizeof(int) * batch_size * max_q_len));
    CHECK(cudaMalloc(&d_cum_seqlens_,     sizeof(int) * (batch_size + 1)));
    CHECK(cudaMalloc(&d_qkv_buf_,         sizeof(T) * token_num * qkv_head_num * head_size_));
    CHECK(cudaMalloc(&d_q_buf_,           sizeof(T) * batch_size * head_num_ * max_q_len * head_size_));
    CHECK(cudaMalloc(&d_k_buf_,           sizeof(T) * batch_size * kv_head_num_ * max_q_len * head_size_));
    CHECK(cudaMalloc(&d_v_buf_,           sizeof(T) * batch_size * kv_head_num_ * max_q_len * head_size_));
    CHECK(cudaMalloc(&d_qk_buf_,          sizeof(T) * batch_size * head_num_ * max_q_len * max_k_len));
    CHECK(cudaMalloc(&d_attn_score_,      sizeof(T) * batch_size * head_num_ * max_q_len * max_k_len));
    CHECK(cudaMalloc(&d_attn_output_,     sizeof(T) * batch_size * head_num_ * max_q_len * head_size_));
    CHECK(cudaMalloc(&d_attn_proj_,       sizeof(T) * token_num * hidden_units_));
    CHECK(cudaMalloc(&d_mask_,            sizeof(T) * batch_size * max_q_len * max_k_len));
    CHECK(cudaMalloc(&d_k_cache_rep_,     sizeof(T) * batch_size * head_num_ * max_k_len * head_size_));
    CHECK(cudaMalloc(&d_v_cache_rep_,     sizeof(T) * batch_size * head_num_ * max_k_len * head_size_));
    CHECK(cudaMalloc(&d_ctx_attn_out_,    sizeof(T) * token_num * hidden_units_));
    CHECK(cudaMalloc(&d_self_qkv_,        sizeof(T) * batch_size * qkv_head_num * head_size_));
    CHECK(cudaMalloc(&d_self_mha_out_,    sizeof(T) * batch_size * hidden_units_));
    CHECK(cudaMalloc(&d_gate_up_,         sizeof(T) * token_num * 2 * inter_size_));
    CHECK(cudaMalloc(&d_ffn_inter_,       sizeof(T) * token_num * inter_size_));
    CHECK(cudaMalloc(&d_ffn_output_,      sizeof(T) * token_num * hidden_units_));
    CHECK(cudaMalloc(&d_all_k_cache_,     sizeof(T) * num_layers_ * batch_size * kv_head_num_ * max_seq_len_ * head_size_));
    CHECK(cudaMalloc(&d_all_v_cache_,     sizeof(T) * num_layers_ * batch_size * kv_head_num_ * max_seq_len_ * head_size_));
    CHECK(cudaMalloc(&d_logits_,          sizeof(T) * batch_size * vocab_size_));
    CHECK(cudaMalloc(&d_topk_ids_,        sizeof(int) * batch_size * 8 * 5));
    CHECK(cudaMalloc(&d_topk_vals_,       sizeof(T) * batch_size * 8 * 5));
    CHECK(cudaMalloc(&d_final_topk_ids_,  sizeof(int) * batch_size * 5));
    CHECK(cudaMalloc(&d_final_topk_vals_, sizeof(T) * batch_size * 5));
    CHECK(cudaMalloc(&d_output_ids_,      sizeof(int) * batch_size));
    CHECK(cudaMalloc(&d_input_lengths_,   sizeof(int) * batch_size));
    CHECK(cudaMalloc(&d_history_lengths_,  sizeof(int) * batch_size));
    CHECK(cudaMalloc(&d_context_lengths_,  sizeof(int) * batch_size));
    CHECK(cudaMalloc(&d_finished_,        sizeof(bool) * batch_size));

    // Zero the KV cache
    CHECK(cudaMemset(d_all_k_cache_, 0, sizeof(T) * num_layers_ * batch_size * kv_head_num_ * max_seq_len_ * head_size_));
    CHECK(cudaMemset(d_all_v_cache_, 0, sizeof(T) * num_layers_ * batch_size * kv_head_num_ * max_seq_len_ * head_size_));
    CHECK(cudaMemset(d_finished_,    0, sizeof(bool) * batch_size));

    DataType dtype = getTensorType<T>();
    DataType int_type = INT32;
    DataType bool_type = BOOL;

    // Create TensorWrappers
    input_ids_buf_      = new TensorWrapper<int>(GPU, int_type, {token_num}, d_input_ids_);
    decoder_input_buf_  = new TensorWrapper<T>(GPU, dtype, {token_num, hidden_units_}, d_decoder_input_);
    decoder_output_buf_ = new TensorWrapper<T>(GPU, dtype, {token_num, hidden_units_}, d_decoder_output_);
    decoder_residual_buf_ = new TensorWrapper<T>(GPU, dtype, {token_num, hidden_units_}, d_decoder_residual_);
    padding_offset_     = new TensorWrapper<int>(GPU, int_type, {batch_size, max_q_len}, d_padding_offset_);
    cum_seqlens_        = new TensorWrapper<int>(GPU, int_type, {batch_size + 1}, d_cum_seqlens_);
    qkv_buf_            = new TensorWrapper<T>(GPU, dtype, {token_num, qkv_head_num, head_size_}, d_qkv_buf_);
    q_buf_              = new TensorWrapper<T>(GPU, dtype, {batch_size, head_num_, max_q_len, head_size_}, d_q_buf_);
    k_buf_              = new TensorWrapper<T>(GPU, dtype, {batch_size, kv_head_num_, max_q_len, head_size_}, d_k_buf_);
    v_buf_              = new TensorWrapper<T>(GPU, dtype, {batch_size, kv_head_num_, max_q_len, head_size_}, d_v_buf_);
    qk_buf_             = new TensorWrapper<T>(GPU, dtype, {batch_size, head_num_, max_q_len, max_k_len}, d_qk_buf_);
    attn_score_buf_     = new TensorWrapper<T>(GPU, dtype, {batch_size, head_num_, max_q_len, max_k_len}, d_attn_score_);
    attn_output_buf_    = new TensorWrapper<T>(GPU, dtype, {batch_size, head_num_, max_q_len, head_size_}, d_attn_output_);
    attn_proj_buf_      = new TensorWrapper<T>(GPU, dtype, {token_num, hidden_units_}, d_attn_proj_);
    mask_buf_           = new TensorWrapper<T>(GPU, dtype, {batch_size, max_q_len, max_k_len}, d_mask_);
    k_cache_repeated_   = new TensorWrapper<T>(GPU, dtype, {batch_size, head_num_, max_k_len, head_size_}, d_k_cache_rep_);
    v_cache_repeated_   = new TensorWrapper<T>(GPU, dtype, {batch_size, head_num_, max_k_len, head_size_}, d_v_cache_rep_);
    ctx_attn_output_    = new TensorWrapper<T>(GPU, dtype, {token_num, hidden_units_}, d_ctx_attn_out_);
    self_qkv_buf_       = new TensorWrapper<T>(GPU, dtype, {batch_size, qkv_head_num, head_size_}, d_self_qkv_);
    self_mha_output_    = new TensorWrapper<T>(GPU, dtype, {batch_size, hidden_units_}, d_self_mha_out_);
    gate_up_buf_        = new TensorWrapper<T>(GPU, dtype, {token_num, 2, inter_size_}, d_gate_up_);
    ffn_inter_buf_      = new TensorWrapper<T>(GPU, dtype, {token_num, inter_size_}, d_ffn_inter_);
    ffn_output_buf_     = new TensorWrapper<T>(GPU, dtype, {token_num, hidden_units_}, d_ffn_output_);
    all_k_cache_        = new TensorWrapper<T>(GPU, dtype, {num_layers_, batch_size, kv_head_num_, max_seq_len_, head_size_}, d_all_k_cache_);
    all_v_cache_        = new TensorWrapper<T>(GPU, dtype, {num_layers_, batch_size, kv_head_num_, max_seq_len_, head_size_}, d_all_v_cache_);
    logits_buf_         = new TensorWrapper<T>(GPU, dtype, {batch_size, vocab_size_}, d_logits_);
    topk_ids_buf_       = new TensorWrapper<int>(GPU, int_type, {batch_size * 8 * 5}, d_topk_ids_);
    topk_vals_buf_      = new TensorWrapper<T>(GPU, dtype, {batch_size * 8 * 5}, d_topk_vals_);
    final_topk_ids_     = new TensorWrapper<int>(GPU, int_type, {batch_size * 5}, d_final_topk_ids_);
    final_topk_vals_    = new TensorWrapper<T>(GPU, dtype, {batch_size * 5}, d_final_topk_vals_);
    output_token_ids_   = new TensorWrapper<int>(GPU, int_type, {batch_size}, d_output_ids_);
    input_lengths_      = new TensorWrapper<int>(GPU, int_type, {batch_size}, d_input_lengths_);
    history_lengths_    = new TensorWrapper<int>(GPU, int_type, {batch_size}, d_history_lengths_);
    context_lengths_    = new TensorWrapper<int>(GPU, int_type, {batch_size}, d_context_lengths_);
    layer_id_           = new TensorWrapper<int>(CPU, int_type, {1}, h_layer_id_);
    step_               = new TensorWrapper<int>(CPU, int_type, {1}, h_step_);
    finished_           = new TensorWrapper<bool>(GPU, bool_type, {batch_size}, d_finished_);
}

template<typename T>
void Llama<T>::freeBuffers() {
    // Free device memory
    if (d_input_ids_)        cudaFree(d_input_ids_);
    if (d_decoder_input_)    cudaFree(d_decoder_input_);
    if (d_decoder_output_)   cudaFree(d_decoder_output_);
    if (d_decoder_residual_) cudaFree(d_decoder_residual_);
    if (d_padding_offset_)   cudaFree(d_padding_offset_);
    if (d_cum_seqlens_)      cudaFree(d_cum_seqlens_);
    if (d_qkv_buf_)          cudaFree(d_qkv_buf_);
    if (d_q_buf_)            cudaFree(d_q_buf_);
    if (d_k_buf_)            cudaFree(d_k_buf_);
    if (d_v_buf_)            cudaFree(d_v_buf_);
    if (d_qk_buf_)           cudaFree(d_qk_buf_);
    if (d_attn_score_)       cudaFree(d_attn_score_);
    if (d_attn_output_)      cudaFree(d_attn_output_);
    if (d_attn_proj_)        cudaFree(d_attn_proj_);
    if (d_mask_)             cudaFree(d_mask_);
    if (d_k_cache_rep_)      cudaFree(d_k_cache_rep_);
    if (d_v_cache_rep_)      cudaFree(d_v_cache_rep_);
    if (d_ctx_attn_out_)     cudaFree(d_ctx_attn_out_);
    if (d_self_qkv_)         cudaFree(d_self_qkv_);
    if (d_self_mha_out_)     cudaFree(d_self_mha_out_);
    if (d_gate_up_)          cudaFree(d_gate_up_);
    if (d_ffn_inter_)        cudaFree(d_ffn_inter_);
    if (d_ffn_output_)       cudaFree(d_ffn_output_);
    if (d_all_k_cache_)      cudaFree(d_all_k_cache_);
    if (d_all_v_cache_)      cudaFree(d_all_v_cache_);
    if (d_logits_)           cudaFree(d_logits_);
    if (d_topk_ids_)         cudaFree(d_topk_ids_);
    if (d_topk_vals_)        cudaFree(d_topk_vals_);
    if (d_final_topk_ids_)   cudaFree(d_final_topk_ids_);
    if (d_final_topk_vals_)  cudaFree(d_final_topk_vals_);
    if (d_output_ids_)       cudaFree(d_output_ids_);
    if (d_input_lengths_)    cudaFree(d_input_lengths_);
    if (d_history_lengths_)  cudaFree(d_history_lengths_);
    if (d_context_lengths_)  cudaFree(d_context_lengths_);
    if (d_finished_)         cudaFree(d_finished_);

    // Free tensor wrappers
    delete input_ids_buf_;       delete decoder_input_buf_;    delete decoder_output_buf_;
    delete decoder_residual_buf_;delete padding_offset_;       delete cum_seqlens_;
    delete qkv_buf_;             delete q_buf_;                delete k_buf_;
    delete v_buf_;               delete qk_buf_;               delete attn_score_buf_;
    delete attn_output_buf_;     delete attn_proj_buf_;        delete mask_buf_;
    delete k_cache_repeated_;    delete v_cache_repeated_;     delete ctx_attn_output_;
    delete self_qkv_buf_;        delete self_mha_output_;      delete gate_up_buf_;
    delete ffn_inter_buf_;       delete ffn_output_buf_;       delete all_k_cache_;
    delete all_v_cache_;         delete logits_buf_;           delete topk_ids_buf_;
    delete topk_vals_buf_;       delete final_topk_ids_;       delete final_topk_vals_;
    delete output_token_ids_;    delete input_lengths_;        delete history_lengths_;
    delete context_lengths_;     delete layer_id_;             delete step_;
    delete finished_;
}

// ========================== Context Decoder (Prefill) ==========================
template<typename T>
void Llama<T>::contextDecode(int* h_input_ids, int* h_input_lengths, int* h_history_lengths,
                              int batch_size, int max_q_len, int max_k_len) {
    // Compute total token count (sum of input lengths)
    int total_tokens = 0;
    for (int i = 0; i < batch_size; i++) {
        total_tokens += h_input_lengths[i];
    }

    // Copy metadata to GPU
    CHECK(cudaMemcpy(d_input_ids_, h_input_ids, sizeof(int) * total_tokens, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_input_lengths_, h_input_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_history_lengths_, h_history_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice));

    // Compute context lengths = history + input
    int h_context_lengths[batch_size];
    for (int i = 0; i < batch_size; i++) {
        h_context_lengths[i] = h_history_lengths[i] + h_input_lengths[i];
    }
    CHECK(cudaMemcpy(d_context_lengths_, h_context_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice));

    // Update tensor shapes for actual dims
    input_ids_buf_->shape = {total_tokens};
    decoder_input_buf_->shape = {total_tokens, hidden_units_};
    decoder_output_buf_->shape = {total_tokens, hidden_units_};
    decoder_residual_buf_->shape = {total_tokens, hidden_units_};
    padding_offset_->shape = {batch_size, max_q_len};
    qkv_buf_->shape = {total_tokens, head_num_ + 2 * kv_head_num_, head_size_};
    q_buf_->shape = {batch_size, head_num_, max_q_len, head_size_};
    k_buf_->shape = {batch_size, kv_head_num_, max_q_len, head_size_};
    v_buf_->shape = {batch_size, kv_head_num_, max_q_len, head_size_};
    mask_buf_->shape = {batch_size, max_q_len, max_k_len};
    qk_buf_->shape = {batch_size, head_num_, max_q_len, max_k_len};
    attn_score_buf_->shape = {batch_size, head_num_, max_q_len, max_k_len};
    attn_output_buf_->shape = {batch_size, head_num_, max_q_len, head_size_};
    k_cache_repeated_->shape = {batch_size, head_num_, max_k_len, head_size_};
    v_cache_repeated_->shape = {batch_size, head_num_, max_k_len, head_size_};
    gate_up_buf_->shape = {total_tokens, 2, inter_size_};
    ffn_inter_buf_->shape = {total_tokens, inter_size_};
    ctx_attn_output_->shape = {total_tokens, hidden_units_};
    attn_proj_buf_->shape = {total_tokens, hidden_units_};
    context_lengths_->shape = {batch_size};

    // 1. Calculate padding offsets
    launchCalPaddingoffset(padding_offset_, cum_seqlens_, input_lengths_);
    CHECK(cudaDeviceSynchronize());

    // 2. Embedding lookup
    launchInputEmbedding(input_ids_buf_, decoder_input_buf_, &weights_->pre_decoder_embedding_weight);
    CHECK(cudaDeviceSynchronize());

    // 3. Build causal mask
    launchBuildCausalMasks(mask_buf_, input_lengths_, context_lengths_);
    CHECK(cudaDeviceSynchronize());

    // Copy embedding output to residual for first layer
    CHECK(cudaMemcpy(d_decoder_residual_, d_decoder_input_, sizeof(T) * total_tokens * hidden_units_, cudaMemcpyDeviceToDevice));

    // 4. Per-layer transformer blocks
    for (int layer = 0; layer < num_layers_; layer++) {
        h_layer_id_[0] = layer;
        auto& layer_weight = *weights_->llama_layer_weight[layer];

        // a. RMSNorm (attention)
        // For first layer, decoder_output = decoder_input (no residual to add)
        // For subsequent layers, decoder_output has the previous layer output
        if (layer == 0) {
            // First layer: just RMSNorm the input
            CHECK(cudaMemcpy(d_decoder_output_, d_decoder_input_,
                           sizeof(T) * total_tokens * hidden_units_, cudaMemcpyDeviceToDevice));
        }
        // Fused add residual + RMSNorm (for layer > 0, residual is already set)
        BaseWeight<T> dummy_norm;
        dummy_norm.bias = nullptr;
        launchFusedAddBiasResidualRMSNorm(decoder_residual_buf_, decoder_output_buf_,
                                           dummy_norm,
                                           layer_weight.attn_norm_weight.gamma,
                                           rmsnorm_eps_);
        CHECK(cudaDeviceSynchronize());

        // b. QKV Linear: [total_tokens, hidden] -> [total_tokens, qkv_head_num, head_size]
        launchLinearGemm(decoder_output_buf_, layer_weight.self_attn_weight.qkv,
                         qkv_buf_, cublas_wrapper);
        CHECK(cudaDeviceSynchronize());

        // c. Fused QKV bias + Transpose + RoPE
        launchAddFusedQKVBiasTransposeAndRoPE(q_buf_, k_buf_, v_buf_, qkv_buf_,
                                               layer_weight.self_attn_weight.qkv,
                                               padding_offset_, history_lengths_, input_lengths_,
                                               attn_static_params_);
        CHECK(cudaDeviceSynchronize());

        // d. Concat KV to cache
        launchConcatKVCache(k_buf_, v_buf_, layer_id_, input_lengths_, history_lengths_,
                            all_k_cache_, all_v_cache_);
        CHECK(cudaDeviceSynchronize());

        // e. Repeat KV cache for GQA (broadcast kv_head_num -> head_num)
        launchRepeatKVCache(all_k_cache_, all_v_cache_, context_lengths_, layer_id_,
                            k_cache_repeated_, v_cache_repeated_);
        CHECK(cudaDeviceSynchronize());

        // f. Q @ K^T: [bs, head_num, q_len, head_size] x [bs, head_num, head_size, k_len]
        //            = [bs, head_num, q_len, k_len]
        launchLinearStridedBatchGemm(q_buf_, k_cache_repeated_, qk_buf_,
                                     cublas_wrapper, false, true);
        CHECK(cudaDeviceSynchronize());

        // g. Scale + Mask + Softmax
        float scale = 1.0f / sqrtf((float)head_size_);
        launchScaleMaskAndSoftmax(qk_buf_, mask_buf_, attn_score_buf_, scale);
        CHECK(cudaDeviceSynchronize());

        // h. Attn_score @ V: [bs, head_num, q_len, k_len] x [bs, head_num, k_len, head_size]
        //                   = [bs, head_num, q_len, head_size]
        launchLinearStridedBatchGemm(attn_score_buf_, v_cache_repeated_, attn_output_buf_,
                                     cublas_wrapper, false, false);
        CHECK(cudaDeviceSynchronize());

        // i. Transpose [bs, head_num, q_len, head_size] -> [total_tokens, hidden_units]
        //    and remove padding
        // attn_output_buf_ shape: [bs, head_num, max_q_len, head_size]
        // ctx_attn_output_ shape: [total_tokens, hidden_units]
        launchTransposeOutRemovePadding(attn_output_buf_, padding_offset_, ctx_attn_output_);
        CHECK(cudaDeviceSynchronize());

        // j. Output linear: [total_tokens, hidden_units] -> [total_tokens, hidden_units]
        attn_proj_buf_->shape = {total_tokens, hidden_units_};
        ctx_attn_output_->shape = {total_tokens, head_num_ * head_size_};
        launchLinearGemm(ctx_attn_output_, layer_weight.self_attn_weight.output,
                         attn_proj_buf_, cublas_wrapper);
        CHECK(cudaDeviceSynchronize());

        // k. Copy attn output to decoder_output for fused add residual + FFN norm
        CHECK(cudaMemcpy(d_decoder_output_, d_attn_proj_,
                        sizeof(T) * total_tokens * hidden_units_, cudaMemcpyDeviceToDevice));

        // l. Fused add residual + RMSNorm (FFN)
        launchFusedAddBiasResidualRMSNorm(decoder_residual_buf_, decoder_output_buf_,
                                           dummy_norm,
                                           layer_weight.ffn_norm_weight.gamma,
                                           rmsnorm_eps_);
        CHECK(cudaDeviceSynchronize());

        // m. Gate+Up Linear: [total_tokens, hidden] -> [total_tokens, 2, inter_size]
        launchLinearGemm(decoder_output_buf_, layer_weight.ffn_weight.gateAndup,
                         gate_up_buf_, cublas_wrapper);
        CHECK(cudaDeviceSynchronize());

        // n. SwiGLU activation: [total_tokens, 2, inter_size] -> [total_tokens, inter_size]
        launchAct(gate_up_buf_, ffn_inter_buf_);
        CHECK(cudaDeviceSynchronize());

        // o. Down Linear: [total_tokens, inter_size] -> [total_tokens, hidden_units]
        ffn_output_buf_->shape = {total_tokens, hidden_units_};
        ffn_inter_buf_->shape = {total_tokens, inter_size_};
        launchLinearGemm(ffn_inter_buf_, layer_weight.ffn_weight.down,
                         ffn_output_buf_, cublas_wrapper);
        CHECK(cudaDeviceSynchronize());

        // p. Set decoder_output for next layer's residual add
        CHECK(cudaMemcpy(d_decoder_output_, d_ffn_output_,
                        sizeof(T) * total_tokens * hidden_units_, cudaMemcpyDeviceToDevice));
    }

    // 5. Final RMSNorm
    BaseWeight<T> dummy_norm_final;
    dummy_norm_final.bias = nullptr;
    launchFusedAddBiasResidualRMSNorm(decoder_residual_buf_, decoder_output_buf_,
                                       dummy_norm_final,
                                       weights_->out_rmsnorm_weight.gamma,
                                       rmsnorm_eps_);
    CHECK(cudaDeviceSynchronize());

    // 6. LM Head: [total_tokens, hidden] -> [total_tokens, vocab_size]
    // For prefill, we only need the last token's logits per sequence
    // But for simplicity, compute all and extract later
    // We need to reshape for the last token only
    // For batch_size=1, token at position total_tokens-1
    TensorWrapper<T> last_token_buf(GPU, getTensorType<T>(),
                                     {batch_size, hidden_units_},
                                     d_decoder_output_ + (total_tokens - batch_size) * hidden_units_);
    logits_buf_->shape = {batch_size, vocab_size_};
    launchLinearGemm(&last_token_buf, weights_->post_decoder_embedding_weight,
                     logits_buf_, cublas_wrapper, false, true);
    CHECK(cudaDeviceSynchronize());
}

// ========================== Self Decoder (Token Generation) ==========================
template<typename T>
void Llama<T>::selfDecode(int* h_input_ids, int* h_history_lengths,
                           int batch_size, int cur_step) {
    // Copy single-token input to GPU
    CHECK(cudaMemcpy(d_input_ids_, h_input_ids, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_history_lengths_, h_history_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    h_step_[0] = cur_step;

    int qkv_head_num = head_num_ + 2 * kv_head_num_;

    // Update shapes for single token
    input_ids_buf_->shape = {batch_size};
    decoder_input_buf_->shape = {batch_size, hidden_units_};
    decoder_output_buf_->shape = {batch_size, hidden_units_};
    decoder_residual_buf_->shape = {batch_size, hidden_units_};
    self_qkv_buf_->shape = {batch_size, qkv_head_num, head_size_};
    self_mha_output_->shape = {batch_size, hidden_units_};
    gate_up_buf_->shape = {batch_size, 2, inter_size_};
    ffn_inter_buf_->shape = {batch_size, inter_size_};
    ffn_output_buf_->shape = {batch_size, hidden_units_};
    logits_buf_->shape = {batch_size, vocab_size_};

    // 1. Embedding lookup for single token
    launchInputEmbedding(input_ids_buf_, decoder_input_buf_, &weights_->pre_decoder_embedding_weight);
    CHECK(cudaDeviceSynchronize());

    // Copy to residual
    CHECK(cudaMemcpy(d_decoder_residual_, d_decoder_input_,
                    sizeof(T) * batch_size * hidden_units_, cudaMemcpyDeviceToDevice));

    // 2. Per-layer transformer
    for (int layer = 0; layer < num_layers_; layer++) {
        h_layer_id_[0] = layer;
        auto& layer_weight = *weights_->llama_layer_weight[layer];

        // a. RMSNorm (attention)
        if (layer == 0) {
            CHECK(cudaMemcpy(d_decoder_output_, d_decoder_input_,
                           sizeof(T) * batch_size * hidden_units_, cudaMemcpyDeviceToDevice));
        }
        BaseWeight<T> dummy_norm;
        dummy_norm.bias = nullptr;
        launchFusedAddBiasResidualRMSNorm(decoder_residual_buf_, decoder_output_buf_,
                                           dummy_norm,
                                           layer_weight.attn_norm_weight.gamma,
                                           rmsnorm_eps_);
        CHECK(cudaDeviceSynchronize());

        // b. QKV Linear: [bs, hidden] -> [bs, qkv_head_num, head_size]
        launchLinearGemm(decoder_output_buf_, layer_weight.self_attn_weight.qkv,
                         self_qkv_buf_, cublas_wrapper);
        CHECK(cudaDeviceSynchronize());

        // c. Fused decoder masked MHA (includes RoPE, KV cache read/write, attention, output)
        launchDecoderMaskedMHA(self_qkv_buf_,
                               layer_weight.self_attn_weight.qkv,
                               layer_id_,
                               all_k_cache_, all_v_cache_,
                               finished_, step_,
                               self_mha_output_,
                               attn_static_params_);
        CHECK(cudaDeviceSynchronize());

        // d. Output linear: [bs, hidden] -> [bs, hidden]
        self_mha_output_->shape = {batch_size, head_num_ * head_size_};
        decoder_output_buf_->shape = {batch_size, hidden_units_};
        launchLinearGemm(self_mha_output_, layer_weight.self_attn_weight.output,
                         decoder_output_buf_, cublas_wrapper);
        CHECK(cudaDeviceSynchronize());

        // Note: at this point decoder_output_buf_ holds the attention output projection
        // We need another buffer or in-place operation for add residual + FFN norm
        // Copy to a temp, then fused add residual + norm writes to decoder_output
        T* d_attn_out_copy = d_ffn_output_; // reuse ffn_output as temp
        CHECK(cudaMemcpy(d_attn_out_copy, d_decoder_output_,
                        sizeof(T) * batch_size * hidden_units_, cudaMemcpyDeviceToDevice));
        decoder_output_buf_->data = d_attn_out_copy;

        // e. Fused add residual + RMSNorm (FFN)
        launchFusedAddBiasResidualRMSNorm(decoder_residual_buf_, decoder_output_buf_,
                                           dummy_norm,
                                           layer_weight.ffn_norm_weight.gamma,
                                           rmsnorm_eps_);
        CHECK(cudaDeviceSynchronize());

        // f. Gate+Up Linear: [bs, hidden] -> [bs, 2, inter_size]
        launchLinearGemm(decoder_output_buf_, layer_weight.ffn_weight.gateAndup,
                         gate_up_buf_, cublas_wrapper);
        CHECK(cudaDeviceSynchronize());

        // g. SwiGLU: [bs, 2, inter_size] -> [bs, inter_size]
        launchAct(gate_up_buf_, ffn_inter_buf_);
        CHECK(cudaDeviceSynchronize());

        // h. Down Linear: [bs, inter_size] -> [bs, hidden]
        ffn_output_buf_->shape = {batch_size, hidden_units_};
        launchLinearGemm(ffn_inter_buf_, layer_weight.ffn_weight.down,
                         ffn_output_buf_, cublas_wrapper);
        CHECK(cudaDeviceSynchronize());

        // Restore decoder_output_buf_ pointer and copy FFN output for next layer
        decoder_output_buf_->data = d_decoder_output_;
        CHECK(cudaMemcpy(d_decoder_output_, d_ffn_output_,
                        sizeof(T) * batch_size * hidden_units_, cudaMemcpyDeviceToDevice));
    }

    // 3. Final RMSNorm
    BaseWeight<T> dummy_norm_final;
    dummy_norm_final.bias = nullptr;
    launchFusedAddBiasResidualRMSNorm(decoder_residual_buf_, decoder_output_buf_,
                                       dummy_norm_final,
                                       weights_->out_rmsnorm_weight.gamma,
                                       rmsnorm_eps_);
    CHECK(cudaDeviceSynchronize());

    // 4. LM Head
    launchLinearGemm(decoder_output_buf_, weights_->post_decoder_embedding_weight,
                     logits_buf_, cublas_wrapper, false, true);
    CHECK(cudaDeviceSynchronize());
}

// ========================== Chat Interface ==========================
template<typename T>
std::vector<std::string> Llama<T>::MakeInput(const std::string &history, int round, const std::string &input) {
    std::vector<std::string> result;
    if (round == 0) {
        result.push_back(input);
    } else {
        result.push_back(history + input);
    }
    return result;
}

template<typename T>
std::string Llama<T>::MakeHistory(const std::string &history, int round,
                                   const std::string &input, const std::string &output) {
    return history + input + output;
}

template<typename T>
std::string Llama<T>::Response(const std::vector<std::string>& input, CallBack PrintRes) {
    // Encode input
    std::string prompt = input[0];
    std::vector<int> input_ids = tokenizer_.Encode(prompt);

    // Prepend BOS token
    input_ids.insert(input_ids.begin(), bos_token_id_);

    int batch_size = 1;
    int input_len = (int)input_ids.size();
    int max_q_len = input_len;
    int max_k_len = input_len; // no history for first call
    int max_gen_len = max_seq_len_ - input_len;

    if (max_gen_len <= 0) {
        std::cout << "[InferSpore] Warning: input too long, no room for generation" << std::endl;
        return "";
    }

    // Allocate buffers
    allocateBuffers(batch_size, max_q_len, max_seq_len_);

    // Prepare host arrays
    int h_input_lengths[1] = {input_len};
    int h_history_lengths[1] = {0};

    // === Prefill ===
    contextDecode(input_ids.data(), h_input_lengths, h_history_lengths,
                  batch_size, max_q_len, max_k_len);

    // Get first generated token via greedy (argmax of top-k)
    topk_ids_buf_->shape = {batch_size * 8 * 5};
    topk_vals_buf_->shape = {batch_size * 8 * 5};
    final_topk_ids_->shape = {batch_size * 5};
    final_topk_vals_->shape = {batch_size * 5};
    logits_buf_->shape = {batch_size, vocab_size_};

    launchTopKforBeamSearch(logits_buf_, topk_ids_buf_, topk_vals_buf_,
                            final_topk_ids_, final_topk_vals_);
    CHECK(cudaDeviceSynchronize());

    // Read top-1 token
    int h_topk_ids[5];
    CHECK(cudaMemcpy(h_topk_ids, d_final_topk_ids_, sizeof(int) * 5, cudaMemcpyDeviceToHost));
    int cur_token = h_topk_ids[0];

    std::string output_text = "";
    std::vector<int> output_ids;

    // === Decode loop ===
    for (int step = input_len + 1; step < max_seq_len_; step++) {
        if (cur_token == eos_token_id_) {
            break;
        }

        output_ids.push_back(cur_token);

        // Emit token via callback (streaming)
        std::string token_str = tokenizer_.Decode({cur_token});
        output_text += token_str;
        if (PrintRes) {
            PrintRes(step - input_len - 1, token_str.c_str());
        }

        // Prepare next step
        int h_next_input[1] = {cur_token};
        int h_next_history[1] = {step - 1}; // all previous positions are history

        // Self-decode one token
        selfDecode(h_next_input, h_next_history, batch_size, step);

        // Sample next token
        launchTopKforBeamSearch(logits_buf_, topk_ids_buf_, topk_vals_buf_,
                                final_topk_ids_, final_topk_vals_);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_topk_ids, d_final_topk_ids_, sizeof(int) * 5, cudaMemcpyDeviceToHost));
        cur_token = h_topk_ids[0];
    }

    // Emit final token if not EOS
    if (cur_token != eos_token_id_) {
        output_ids.push_back(cur_token);
        std::string token_str = tokenizer_.Decode({cur_token});
        output_text += token_str;
        if (PrintRes) {
            PrintRes((int)output_ids.size() - 1, token_str.c_str());
        }
    }

    freeBuffers();
    return output_text;
}

// ========================== Template Instantiation ==========================
template class Llama<float>;
template class Llama<half>;
