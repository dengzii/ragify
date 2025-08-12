//
// Created by dengzi on 2025/8/12.
//

#ifndef C_API_H
#define C_API_H

#ifdef __cplusplus
extern "C" {
#endif

int load_embedding_model(char *model_path);

int load_rerank_model(char *model_path);

#ifdef __cplusplus
}
#endif

#endif //C_API_H
