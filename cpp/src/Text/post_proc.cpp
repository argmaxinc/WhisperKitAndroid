//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#include "post_proc.hpp"

#define LOGITS_TO_NEG_INF(start, end) \
    for (auto iter = (start); iter != (end); iter++) *iter = -1e9;

#define DEC_2_ROUND(x) (round((x)*100.0) / 100.0)

using namespace std;
using json = nlohmann::json;

PostProcModel::PostProcModel(Tokenizer* tokenizer, bool timestamp_text) : MODEL_SUPER_CLASS("post_proc") {
    _timestamp_text = timestamp_text;
    _tokenizer = tokenizer;
}

bool PostProcModel::initialize(bool debug) {
    if (!MODEL_SUPER_CLASS::initializeModelInMemory(WhisperKit::InMemoryModel::ModelType::kSimplePostProcessingModel,
                                                    debug)) {
        LOGE("Failed to initialize\n");
        return false;
    }
    return true;
}

void PostProcModel::invoke(bool measure_time) { MODEL_SUPER_CLASS::invoke(measure_time); }

void PostProcModel::apply_timestamp_rules(float* logits, int logits_size, vector<int>& tokens) {
    logits[_tokenizer->specialTokens.noTimestampsToken] = -1e9;
    bool last_was_timestamp =
        (tokens.size() >= 2 && tokens[tokens.size() - 1] >= _tokenizer->specialTokens.timestampBeginToken);
    bool penultimate_was_timestamp =
        (tokens.size() < 3 || tokens[tokens.size() - 2] >= _tokenizer->specialTokens.timestampBeginToken);

    if (last_was_timestamp) {
        if (penultimate_was_timestamp)  // has to be non-timestamp
            LOGITS_TO_NEG_INF(&logits[_tokenizer->specialTokens.timestampBeginToken], &logits[logits_size])
        else  // cannot be normal text tokens
            LOGITS_TO_NEG_INF(logits, &logits[_tokenizer->specialTokens.endOfTranscriptToken])
    }

    vector<int> timestamps;
    int timestamp_last;
    for (auto& token : tokens) {
        if (token >= _tokenizer->specialTokens.timestampBeginToken) timestamps.push_back(token);
    }
    if (timestamps.size() > 0) {
        if (last_was_timestamp && !penultimate_was_timestamp)
            timestamp_last = timestamps[timestamps.size() - 1];
        else
            timestamp_last = timestamps[timestamps.size() - 1] + 1;

        LOGITS_TO_NEG_INF(&logits[_tokenizer->specialTokens.timestampBeginToken], &logits[timestamp_last])
    }

    if (tokens.size() == SAMPLE_BEGIN) {
        LOGITS_TO_NEG_INF(logits, &logits[_tokenizer->specialTokens.timestampBeginToken])

        auto max_initial_timestamp_index = int(1.0 / 0.02);
        int last_allowed = _tokenizer->specialTokens.timestampBeginToken + max_initial_timestamp_index;
        LOGITS_TO_NEG_INF(&logits[last_allowed + 1], &logits[logits_size])
    }
}

int PostProcModel::process(int idx, float* logits, int logits_size, vector<int>& decoded_tokens, float base_timestamp) {
    chrono::time_point<chrono::high_resolution_clock> before_exec = chrono::high_resolution_clock::now();

    if (idx == 0) {
        logits[_tokenizer->specialTokens.endOfTranscriptToken] = -1e9;
        logits[_tokenizer->specialTokens.blankToken] = -1e9;
    }
    for (int i = 0; i < _tokenizer->numNonSpeechTokens; i++) {
        auto token = _tokenizer->nonSpeechTokens[i];
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

    auto timestamp_logprob = reinterpret_cast<float*>(outputs[0].first)[0];
    auto max_text_token_logprob = reinterpret_cast<float*>(outputs[1].first)[0];
    if (timestamp_logprob > max_text_token_logprob) {
        LOGITS_TO_NEG_INF(logits, &logits[_tokenizer->specialTokens.timestampBeginToken])
    }

    vector<float> v_logits(logits, &logits[logits_size]);
    auto max_elem = max_element(v_logits.begin(), v_logits.end());

    auto after_exec = chrono::high_resolution_clock::now();
    float interval_infs = chrono::duration_cast<std::chrono::microseconds>(after_exec - before_exec).count() / 1000.0;
    _latencies.push_back(interval_infs);
    auto token = distance(v_logits.begin(), max_elem);

    return token;
}

void PostProcModel::decode_segment(const std::vector<int>& tokens) {
    char* c_word = tokenizer_decode(_tokenizer, tokens.data(), tokens.size(), false);
    _sentence += std::string(c_word);
    tokenizer_free_rstring(c_word);
}

void PostProcModel::proc_token(int token, float base_timestamp) {
    if (token == _tokenizer->specialTokens.endOfTranscriptToken) return;

    if (token == _tokenizer->specialTokens.blankToken) return;

    char* c_word = tokenizer_decode(_tokenizer, &token, 1, false);

    string word, timestr;
    size_t start, end;
    word = std::string(c_word);

    if (token >= _tokenizer->specialTokens.startOfTranscriptToken) {
        start = word.find("<|");
        end = word.find("|>");

        if (start != string::npos && end != string::npos && end > start) {
            if (_timestamp_text) {
                start += 2;
                timestr = word.substr(start, end - start);
                if (timestr.find_first_not_of("0123456789.", start) != string::npos) return;

                // this is optional. Whisper encoder doesn't support timestamps beyong
                // 30sec mark, but I add the previous segment time to the latest timestamp
                // so it's easier to track the entire time in the transcript
                auto timestamp = (int)(stof(timestr) * 100) + (int)(base_timestamp * 100);
                string ts_str = to_string(timestamp / 100.0);
                ts_str.erase(ts_str.find_last_not_of('0') + 1, std::string::npos);
                ts_str.erase(ts_str.find_last_not_of('.') + 1, std::string::npos);
                word = "<|" + ts_str + "|>";
            } else {
                word.erase(start, end + 2);
            }
        }
    }

    _sentence += word;

    tokenizer_free_rstring(c_word);
}

unique_ptr<string> PostProcModel::get_sentence(bool bClear) {
    auto out = make_unique<string>(_sentence);
    if (bClear) _sentence.clear();

    return out;
}
