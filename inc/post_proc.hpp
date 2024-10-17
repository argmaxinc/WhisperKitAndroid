//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#pragma once

#include <cmath>
#include <memory>
#include <tflite_model.hpp>
#include <vector>

#define TOKEN_SOT 50257  // Start of transcript
#define TOKEN_EOT 50256  // end of transcript
#define TOKEN_BLANK 220  // " "
#define TOKEN_NO_TIMESTAMP 50362
#define TOKEN_TIMESTAMP_BEGIN 50363
#define TOKEN_NO_SPEECH 50361
#define SAMPLE_BEGIN 1
#define NO_SPEECH_THR 0.6

class PostProcModel : public TFLiteModel {
   public:
    PostProcModel(const string& token_file);
    virtual ~PostProcModel() {};

    bool initialize(string model_path, string lib_dir, string cache_dir, int backend = kUndefinedBackend);
    virtual void invoke(bool measure_time = false);

    int process(int idx, float* logits, int logits_size, vector<int>& decoded_tokens, float base_timestamp);

    unique_ptr<string> get_sentence(bool clear = true);

   private:
    string _token_file;
    unique_ptr<nlohmann::json> _vocab_json;
    string _sentence;

    void apply_timestamp_rules(float* logits, int logits_size, vector<int>& tokens);
    void proc_token(int token, float base_timestamp);
};
