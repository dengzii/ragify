#include <algorithm>
#include <chrono>
#include <iostream>

#include "../src/ragify.h"


static float similarity(const std::vector<float> &emb1, const std::vector<float> &emb2) {
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

static void query_documents(const ragify &r, const std::string &query,
                            const std::vector<std::string> &documents) {
    const auto now = std::chrono::high_resolution_clock::now();
    const auto embdQuery = r.get_embeddings({query})[0];
    const auto embeddings = r.get_embeddings(documents);
    const auto elapsed = std::chrono::high_resolution_clock::now() - now;

    std::cout << "embedding time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms"
            << std::endl;

    std::vector<std::pair<float, std::string> > results;

    for (size_t i = 0; i < embeddings.size(); i++) {
        const auto &embdText = embeddings[i];
        const auto similarityScore = similarity(embdQuery, embdText);
        results.emplace_back(similarityScore, documents[i]);
    }

    std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) { return a.first > b.first; });

    std::cout << "query: " << query << std::endl;
    for (const auto &[fst, snd]: results) {
        std::cout << fst << " => " << snd << std::endl;
    }
}

static void rerank(const ragify &r, const std::string &query, const std::vector<std::string> &documents) {
    const auto scores = r.rerank(query, documents);

    std::vector<std::pair<float, std::string> > results;
    for (size_t i = 0; i < documents.size(); i++) {
        results.emplace_back(scores[i], documents[i]);
    }
    std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

    std::cout << "query: " << query << std::endl;
    for (const auto item: results) {
        std::cout << item.first << " => " << item.second << std::endl;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    const std::vector<std::string> documents = {
        "an apple a day keeps the doctor away",
        "cat and a dog are friends",
        "banana has a high potassium content",
        "i need medicine to cure a cold",
        "do not eat too much salt",
        "compute games make me happy",
        "pets need vaccine shots",
        "walking is good for health",
        "tom like to talk with his friends",
    };

    const std::string query = "how to keep healthy";

    const auto model_path = std::string(argv[5]);

    auto r = ragify();

    r.load_model(model_path);
    query_documents(r, query, documents);

    // r.load_rerank_model(model_path);
    // rerank(r, query, documents);

    return 0;
}
