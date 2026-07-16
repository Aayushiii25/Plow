#pragma once
#include <string>

struct LLaMAAttentionStaticParams {
    int   rotary_embedding_dim = 128;
    float rotary_embedding_base = 10000.0f;
    int   max_position_embeddings = 2048;
    bool  use_dynamic_ntk = false;
};

struct LlamaModelParams {
    int     head_num = 32;
    int     kv_head_num = 32;
    int     head_size = 128;
    int     hidden_units = 4096;       // head_num * head_size
    int     inter_size = 11008;
    int     num_layers = 32;
    int     vocab_size = 32000;
    int     max_seq_len = 2048;
    float   rmsnorm_eps = 1e-5f;
    bool    attn_bias = false;
    int     bos_token_id = 1;
    int     eos_token_id = 2;
    int     pad_token_id = 0;

    LLaMAAttentionStaticParams attn_params;

    void derive() {
        hidden_units = head_num * head_size;
        attn_params.rotary_embedding_dim = head_size;
    }
};
