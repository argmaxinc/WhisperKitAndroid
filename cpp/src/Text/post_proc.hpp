//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "Tokenizer.h"
#include "backend_class.hpp"

constexpr const uint32_t SAMPLE_BEGIN = 1;

class PostProcModel : public MODEL_SUPER_CLASS {
   public:
    PostProcModel(Tokenizer* tokenizer, bool timestamp_text = false);
    virtual ~PostProcModel() {};

    bool initialize(bool debug = false);
    virtual void invoke(bool measure_time = false);

    int process(int idx, float* logits, int logits_size, std::vector<int>& decoded_tokens, float base_timestamp);

    std::unique_ptr<std::string> get_sentence(bool clear = true);
    void decode_segment(const std::vector<int>& tokens);

   private:
    Tokenizer* _tokenizer;
    bool _timestamp_text;
    std::string _sentence;

    void apply_timestamp_rules(float* logits, int logits_size, std::vector<int>& tokens);
    void proc_token(int token, float base_timestamp);
};
