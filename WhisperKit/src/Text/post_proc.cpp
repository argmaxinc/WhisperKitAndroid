//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#include "post_proc.hpp"

#define LOGITS_TO_NEG_INF(start, end) \
        for(auto iter = (start); iter != (end); iter++) \
            *iter = -1e9;

#define DEC_2_ROUND(x) (round((x) * 100.0) / 100.0)


using namespace std;
using json = nlohmann::json;

PostProcModel::PostProcModel(
    const string& token_file, bool timestamp_text
):MODEL_SUPER_CLASS("post_proc")
{
    _token_file = token_file;
    _timestamp_text = timestamp_text;
}

bool PostProcModel::initialize(
    string model_file, 
    string lib_dir,
    string cache_dir, 
    int backend,
    bool debug
){
    ifstream json_file(_token_file);
    _vocab_json = make_unique<json>(json::parse(json_file));

    LOGI("postproc vocab size: %lu\n", _vocab_json->size());

    if(!MODEL_SUPER_CLASS::initialize(model_file, lib_dir, cache_dir, backend, debug)){
        LOGE("Failed to initialize\n");
        return false;
    }
    return true;
}

void PostProcModel::invoke(bool measure_time){
    MODEL_SUPER_CLASS::invoke(measure_time);
}

void PostProcModel::apply_timestamp_rules(
    float* logits, 
    int logits_size, 
    vector<int>& tokens
){
    logits[TOKEN_NO_TIMESTAMP] = -1e9;
    bool last_was_timestamp = (tokens.size() >= 2 && 
        tokens[tokens.size() - 1] >= TOKEN_TIMESTAMP_BEGIN
    );
    bool penultimate_was_timestamp = (tokens.size() < 3 || 
        tokens[tokens.size() - 2] >= TOKEN_TIMESTAMP_BEGIN
    );

    if (last_was_timestamp){
        if (penultimate_was_timestamp)  // has to be non-timestamp
            LOGITS_TO_NEG_INF( 
                &logits[TOKEN_TIMESTAMP_BEGIN], 
                &logits[logits_size]
            )
        else                            // cannot be normal text tokens
            LOGITS_TO_NEG_INF( 
                logits, 
                &logits[TOKEN_EOT]
            )
    }

    vector<int> timestamps;
    int timestamp_last;
    for(auto& token : tokens){
        if (token >= TOKEN_TIMESTAMP_BEGIN)
            timestamps.push_back(token);
    }
    if(timestamps.size() > 0){
        if(last_was_timestamp && !penultimate_was_timestamp)
            timestamp_last = timestamps[timestamps.size() - 1];
        else
            timestamp_last = timestamps[timestamps.size() - 1] + 1;

        LOGITS_TO_NEG_INF( 
            &logits[TOKEN_TIMESTAMP_BEGIN], 
            &logits[timestamp_last]
        )
    }

    if(tokens.size() == SAMPLE_BEGIN){
        LOGITS_TO_NEG_INF( 
            logits, 
            &logits[TOKEN_TIMESTAMP_BEGIN]
        )

        auto max_initial_timestamp_index = int(1.0 / 0.02);
        int last_allowed = TOKEN_TIMESTAMP_BEGIN + max_initial_timestamp_index;
        LOGITS_TO_NEG_INF( 
            &logits[last_allowed + 1], 
            &logits[logits_size]
        )
    }
}

int PostProcModel::process(
    int idx, 
    float* logits, 
    int logits_size,
    vector<int>& decoded_tokens, 
    float base_timestamp
){
    chrono::time_point<chrono::high_resolution_clock> 
        before_exec = chrono::high_resolution_clock::now();

    if (idx == 0) {
        logits[TOKEN_EOT] = -1e9;
        logits[TOKEN_BLANK] = -1e9;
    }
    for(auto& token : _non_speech_tokens){
        logits[token] = -1e9;
    }
    apply_timestamp_rules(logits, logits_size, decoded_tokens);
    // logits
    read_input_data(reinterpret_cast<char*>(logits), 0);
    auto inputs = get_input_ptrs();

    invoke();

    auto outputs = get_output_ptrs();
    /* outputs: outputs[0] is timestamp_logprob, 
     *          outputs[1] is max_text_token_logprob, 
     *          outputs[2] is logprobs[TOKEN_NO_SPEECH]
     */
    if(outputs.size() != 1)
        throw std::invalid_argument("outputs.size should be 1");


    auto* logprobs = reinterpret_cast<float*>(outputs[0].first);
    auto timestamp_logprob = logprobs[0];
    auto max_text_token_logprob = logprobs[1];
    if(timestamp_logprob > max_text_token_logprob)
    {
        LOGITS_TO_NEG_INF( 
            logits, 
            &logits[TOKEN_TIMESTAMP_BEGIN]
        )
    }

    vector<float> v_logits(logits, &logits[logits_size]);
    auto max_elem = max_element(v_logits.begin(), v_logits.end());

    auto after_exec = chrono::high_resolution_clock::now();
    float interval_infs =
        chrono::duration_cast<std::chrono::microseconds>(
            after_exec - before_exec).count() / 1000.0;
    _latencies.push_back(interval_infs);
    auto token = distance(v_logits.begin(), max_elem);
    proc_token(token, base_timestamp);

    return token;
}

void PostProcModel::proc_token(int token, float base_timestamp)
{
    if(token == TOKEN_EOT)
        return;
    string word, timestr;
    size_t start, end;

    word = (*_vocab_json)[to_string(token)];
    if(token == TOKEN_BLANK)
        return;

    else if (token >= TOKEN_SOT){
        start = word.find("<|");
        end = word.find("|>");
        
        if( start != string::npos && end != string::npos && end > start){
            if (_timestamp_text){
                start += 2;
                timestr = word.substr(start, end-start);
                if(timestr.find_first_not_of("0123456789.", start) != string::npos)
                    return;

                // this is optional. Whisper encoder doesn't support timestamps beyong
                // 30sec mark, but I add the previous segment time to the latest timestamp
                // so it's easier to track the entire time in the transcript
                auto timestamp = (int)(stof(timestr) * 100) + (int)(base_timestamp * 100);
                string ts_str = to_string(timestamp / 100.0);
                ts_str.erase ( ts_str.find_last_not_of('0') + 1, std::string::npos );
                ts_str.erase ( ts_str.find_last_not_of('.') + 1, std::string::npos );
                word = "<|" + ts_str + "|>";
            } else {
                word.erase(start, end + 2);
            }
        }
    }
    if (word.find_first_of(" .,'\"?!") != string::npos)
        _sentence += word;
    else
        _sentence += " " + word;
}

unique_ptr<string> PostProcModel::get_sentence(bool bClear)   
{ 
    auto out = make_unique<string>(_sentence);
    if(bClear)
        _sentence.clear();

    return out; 
}
