//
// Created by dengzi on 2025/8/11.
//

#ifndef RAG_H
#define RAG_H

#include <string>
#include <vector>
#include "llama.h"

struct params {
    std::string rerank_model_path;
    std::string embedding_model_path;

    int chunk_size_min = 128;
    int chunk_size_max = 256;
};

class ragify {
public:
    explicit ragify();

    int load_model(const std::string &model_path);

    int load_rerank_model(const std::string &model_path);

    [[nodiscard]] std::vector<std::vector<float> > get_embeddings(const std::vector<std::string> &inputs) const;

    [[nodiscard]] std::vector<float> rerank(const std::string &query, const std::vector<std::string> &documents) const;

    void release();

    static float similarity(const std::vector<float> &emb1, const std::vector<float> &emb2);

private:
    llama_model *model;
    llama_context *ctx;
    int embd_normalize_type = 2;

    llama_model *model_rerank;
    llama_context *ctx_rerank;

    static int batch_decode(llama_context *ctx, const llama_batch &batch, float *output, int n_embd, int normalization);

    [[nodiscard]] float rank_document(const std::string &query, const std::string &document) const;
};


#endif //RAG_H
