//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#include <memory>
#include <cmath>
#include <vector>

#include "backend_class.hpp"

constexpr const uint32_t TOKEN_SOT = 50257;  // Start of transcript
constexpr const uint32_t TOKEN_EOT = 50256;  // end of transcript
constexpr const uint32_t TOKEN_BLANK = 220;  // " "
constexpr const uint32_t TOKEN_NO_TIMESTAMP = 50362;
constexpr const uint32_t TOKEN_TIMESTAMP_BEGIN = 50363;
constexpr const uint32_t TOKEN_NO_SPEECH = 50361;
constexpr const uint32_t SAMPLE_BEGIN = 1;

class PostProcModel: public MODEL_SUPER_CLASS{
public:
    PostProcModel(const std::string& token_file, bool timestamp_text=false);
    virtual ~PostProcModel() {};

    bool initialize(
        bool debug=false
    );
    virtual void invoke(bool measure_time=false);

    int process(
        int idx, 
        float* logits, 
        int logits_size,
        std::vector<int>& decoded_tokens,
        float base_timestamp
    );

    std::unique_ptr<std::string> get_sentence(bool clear=true);

private:
    bool _timestamp_text;
    std::string _token_file;
    std::unique_ptr<nlohmann::json> _vocab_json;
    std::string _sentence;
    const std::array<int, 90> _non_speech_tokens  = {
        1,      2,      7,      8,      9,      10,     14,     25, 
        26,     27,     28,     29,     31,     58,     59,     60, 
        61,     62,     63,     90,     91,     92,     93,     357,
        366,    438,    532,    685,    705,    796,    930,    1058,
        1220,   1267,   1279,   1303,   1343,   1377,   1391,   1635, 
        1782,   1875,   2162,   2361,   2488,   3467,   4008,   4211,   
        4600,   4808,   5299,   5855,   6329,   7203,   9609,   9959,
        10563,  10786,  11420,  11709,  11907,  13163,  13697,  13700,
        14808,  15306,  16410,  16791,  17992,  19203,  19510,  20724,
        22305,  22935,  27007,  30109,  30420,  33409,  34949,  40283,
        40493,  40549,  47282,  49146,  50257,  50357,  50358,  50359,
        50360,  50361,
    };

    void apply_timestamp_rules(    
        float* logits, 
        int logits_size, 
        std::vector<int>& tokens
    );
    void proc_token(int token, float base_timestamp);
};
