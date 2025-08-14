#include "ragify.h"

#include <cmath>
#include <cstdarg>
#include <iostream>
#include "llama-context.h"
#include "llama-model.h"
#include "common.h"

static void LOGE(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    printf("\n");
}

static void LOGI(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}


struct result {
    std::string text;
    std::vector<llama_token> tokens;
    std::vector<float> embedding;
};

ragify::ragify(): model(nullptr), ctx(nullptr),
                  model_rerank(nullptr), ctx_rerank(nullptr) {
}


int ragify::load_model(const std::string &model_path) {
    release();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        release();
        return 1;
    }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.no_perf = true;
    // ctx_params.n_ctx = 32768;

    ctx = llama_init_from_model(model, ctx_params);
    LOGI("embedding model loaded: %s, pooling_type: %d, dim: %d", model->name.c_str(), ctx->pooling_type(),
         llama_model_n_embd(model));
    if (!ctx) {
        release();
        return 2;
    }
    return 0;
}


int ragify::load_rerank_model(const std::string &model_path) {
    llama_model_params lparams = llama_model_default_params();
    lparams.n_gpu_layers = 0;
    model_rerank = llama_model_load_from_file(model_path.c_str(), lparams);
    if (!model_rerank) {
        LOGE("Failed to load rerank model");
        return 1;
    }

    llama_context_params params = llama_context_default_params();
    params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    params.embeddings = true;
    // params.n_batch = 1024;
    // params.n_ctx = 2048;

    ctx_rerank = llama_init_from_model(model_rerank, params);
    if (!ctx_rerank) {
        LOGE("Failed to initialize rerank context");
        llama_model_free(model_rerank);
        model_rerank = nullptr;
        return 2;
    }
    LOGI("rerank model loaded: %s", model_rerank->name.c_str());
    return 0;
}

void ragify::release() {
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_model_free(model);
        LOGI("release embedding model");
    }
    if (ctx_rerank) {
        llama_free(ctx_rerank);
    }
    if (model_rerank) {
        llama_model_free(model_rerank);
        LOGI("release rerank model");
    }
    ctx = nullptr;
    model = nullptr;
    ctx_rerank = nullptr;
    model_rerank = nullptr;
}

int ragify::batch_decode(llama_context *ctx, const llama_batch &batch, float *output, const int n_embd,
                         const int normalization) {
    if (!output) {
        LOGE("output is null");
        return -1;
    }

    if (llama_decode(ctx, batch) < 0) {
        LOGE("failed to decode");
        return -1;
    }
    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }
        const float *embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == nullptr) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == nullptr) {
                LOGE("failed to get embeddings");
                return -1;
            }
        }
        float *out = output + batch.seq_id[i][0] * n_embd;
        common_embd_normalize(embd, out, n_embd, normalization);
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    return 0;
}

std::vector<std::vector<float> > ragify::get_embeddings(const std::vector<std::string> &inputs) const {
    if (!model) {
        LOGE("model not loaded");
        return {};
    }
    if (!ctx) {
        LOGE("context not initialized");
        return {};
    }

    std::vector<result> results;

    // max batch size
    const auto n_batch = static_cast<int>(llama_n_batch(ctx));

    for (auto &input: inputs) {
        auto tokens = common_tokenize(ctx, input, true, false);
        if (n_batch < static_cast<int>(tokens.size())) {
            LOGE("input size exceeds max batch size");
            return {};
        }
        results.push_back(result{input, tokens, {}});
    }

    const auto n_seq_max = static_cast<int>(llama_n_seq_max(ctx));
    const auto n_embd = llama_model_n_embd(model);
    const auto input_size = results.size();

    llama_batch batch = llama_batch_init(n_batch, 0, n_seq_max);

    std::vector<float> embeddings(input_size * n_embd, 0);
    float *emb = embeddings.data();

    int p = 0; // number of prompts processed already
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < input_size; k++) {
        // clamp to n_batch tokens
        auto &inp = results[k].tokens;
        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch) {
            float *out = emb + p * n_embd;
            if (const auto r = batch_decode(ctx, batch, out, n_embd, 2); r < 0) {
                LOGE("failed to decode batch");
                return {};
            }
            batch.n_tokens = 0;
            p += s;
            s = 0;
        }
        const size_t n_tokens = inp.size();
        for (size_t i = 0; i < n_tokens; i++) {
            batch.token[batch.n_tokens] = inp[i];
            batch.pos[batch.n_tokens] = static_cast<int>(i);
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = s;
            batch.logits[batch.n_tokens] = true;
            batch.n_tokens++;
        }
        s += 1;
    }

    float *out = emb + p * n_embd;
    const auto ret = batch_decode(ctx, batch, out, n_embd, 2);
    if (ret < 0) {
        LOGE("failed to decode batch");
        return {};
    }

    std::vector<std::vector<float> > result_embeddings;
    for (int i = 0; i < input_size; i++) {
        results[i].embedding = std::vector<float>(emb + i * n_embd, emb + (i + 1) * n_embd);
        results[i].tokens.clear();
        result_embeddings.push_back(results[i].embedding);
    }
    llama_batch_free(batch);

    return result_embeddings;
}

float ragify::similarity(const std::vector<float> &emb1, const std::vector<float> &emb2) {
    if (emb1.size() != emb2.size() || emb1.empty()) {
        return 0.0f;
    }

    double sum = 0.0, sum1 = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < emb1.size(); i++) {
        sum += emb1[i] * emb2[i];
        sum1 += emb1[i] * emb1[i];
        sum2 += emb2[i] * emb2[i];
    }

    if (sum1 == 0.0 || sum2 == 0.0) {
        return (sum1 == 0.0 && sum2 == 0.0) ? 1.0f : 0.0f;
    }

    return static_cast<float>(sum / (std::sqrt(sum1) * std::sqrt(sum2)));
}

std::vector<llama_token> truncate_tokens(const std::vector<llama_token> &tokens, int limit_size,
                                         llama_token eos_token) {
    std::vector<llama_token> new_tokens = tokens;

    if ((int) tokens.size() > limit_size) {
        LOGE("token size exceeds limit");
        new_tokens.resize(limit_size);
    }
    // add eos if not present
    if (new_tokens.empty() || new_tokens.back() != eos_token) {
        new_tokens.push_back(eos_token);
    }
    return new_tokens;
}


float ragify::rank_document(const std::string &query, const std::string &document) const {
    if (!model_rerank || !ctx_rerank) {
        LOGE("rerank model not loaded");
        return 0.0f;
    }

    std::vector<llama_token> tokens;

    const auto n_batch = llama_n_batch(ctx_rerank);
    const auto *vocab = llama_model_get_vocab(model_rerank);
    const auto tokenBos = llama_vocab_bos(vocab);
    const auto tokenEos = llama_vocab_eos(vocab);
    const auto tokenSep = llama_vocab_sep(vocab);

    tokens.reserve(n_batch);

    const auto maxSize = static_cast<int>(n_batch / 2) - 2;

    tokens.push_back(tokenBos);

    auto part1 = common_tokenize(ctx_rerank, query, false);
    if (part1.size() > maxSize) {
        LOGE("token size exceeds limit");
        part1.resize(maxSize);
    }
    tokens.insert(tokens.end(), part1.begin(), part1.end());
    tokens.push_back(tokenEos);
    tokens.push_back(tokenSep);

    auto part2 = common_tokenize(ctx_rerank, document, false);
    if (part2.size() > maxSize) {
        LOGE("token size exceeds limit");
        part2.resize(maxSize);
    }
    tokens.insert(tokens.end(), part2.begin(), part2.end());
    tokens.push_back(tokenEos);

    //////////////

    const uint32_t n_ctx = ctx_rerank->n_ctx();
    const int n_embd = llama_model_n_embd(model_rerank);

    if (tokens.size() > n_ctx) {
        LOGE("input size exceeds max context");
        return 0.0f;
    }

    llama_batch batch = llama_batch_init(static_cast<int32_t>(n_batch), 0, 1);

    for (llama_pos i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], i, {0}, true);
    }

    if (llama_encode(this->ctx_rerank, batch) < 0) {
        LOGE("llama_encode failed");
        llama_batch_free(batch);
        return 0.0f;
    }

    std::vector embeddings(n_embd, 0.0f);

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!batch.logits[i]) {
            continue;
        }
        const float *embd = llama_get_embeddings_seq(this->ctx_rerank, batch.seq_id[i][0]);
        if (embd == nullptr) {
            embd = llama_get_embeddings_ith(this->ctx_rerank, i);
        }
        if (embd == nullptr) {
            LOGE("Failed to get embeddings");
            continue;
        }
        common_embd_normalize(embd, embeddings.data(), n_embd, -1);
        break;
    }

    llama_batch_free(batch);

    return embeddings.at(0);
}


std::vector<float> ragify::rerank(const std::string &query, const std::vector<std::string> &documents) const {
    std::vector<float> scores;
    scores.reserve(documents.size());
    for (const std::string &doc: documents) {
        scores.push_back(rank_document(query, doc));
    }
    return scores;
}
