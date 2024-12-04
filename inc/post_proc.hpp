//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#include <memory>
#include <cmath>
#include <vector>

#include <backend_class.hpp>

#define TOKEN_SOT 50257  // Start of transcript
#define TOKEN_EOT 50256  // end of transcript
#define TOKEN_BLANK 220  // " "
#define TOKEN_NO_TIMESTAMP 50362
#define TOKEN_TIMESTAMP_BEGIN 50363
#define TOKEN_NO_SPEECH 50361
#define SAMPLE_BEGIN 1
#define NO_SPEECH_THR 0.6

class PostProcModel: public MODEL_SUPER_CLASS{
public:
    PostProcModel(const string& token_file, bool timestamp_text=false);
    virtual ~PostProcModel() {};

    bool initialize(
        string model_path, 
        string lib_dir,
        string cache_dir, 
        int backend, 
        bool debug=false
    );
    virtual void invoke(bool measure_time=false);

    int process(
        int idx, 
        float* logits, 
        int logits_size,
        vector<int>& decoded_tokens,
        float base_timestamp
    );

    unique_ptr<string> get_sentence(bool clear=true);

private:
    bool _timestamp_text;
    string _token_file;
    unique_ptr<nlohmann::json> _vocab_json;
    string _sentence;

    void apply_timestamp_rules(    
        float* logits, 
        int logits_size, 
        vector<int>& tokens
    );
    void proc_token(int token, float base_timestamp);
};
