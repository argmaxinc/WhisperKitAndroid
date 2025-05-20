#include "Tokenizer.h"

#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "stddef.h"
#include "stdlib.h"
#include "tflite_msg.hpp"

using json = nlohmann::json;
#ifdef __cplusplus
extern "C" {
#endif

typedef struct CEncoding CEncoding;

struct TokenizerHandle *tokenizer_from_file(const char *path);

char *tokenizer_decode(TokenizerHandle *handle, const unsigned int *ids, size_t length, bool skip_special_tokens);

struct CEncoding *tokenizer_encode(struct TokenizerHandle *handle, const char *text, bool add_special_tokens);
uintptr_t encoding_get_length(const struct CEncoding *encoding);
const uint32_t *encoding_get_ids(const struct CEncoding *encoding, uintptr_t *length);

void free_encoding_array(struct CEncoding *array, uintptr_t length);

void tokenizer_free(struct TokenizerHandle *handle);

void free_rstring(char *s);

#ifdef __cplusplus
}
#endif

void init_special_tokens(Tokenizer *tokenizer) {
    int sot_token = tokenizer_convert_token_to_id(tokenizer, "<|startoftranscript|>");
    int eot_token = tokenizer_convert_token_to_id(tokenizer, "<|endoftext|>");
    int blank_token = tokenizer_convert_token_to_id(tokenizer, " ");
    int transcribe_token = tokenizer_convert_token_to_id(tokenizer, "<|transcribe|>");
    int translate_token = tokenizer_convert_token_to_id(tokenizer, "<|translate|>");
    int timestamp_begin_token = tokenizer_convert_token_to_id(tokenizer, "<|0.00|>");
    int english_token = tokenizer_convert_token_to_id(tokenizer, "<|en|>");
    int no_speech_token = tokenizer_convert_token_to_id(tokenizer, "<|nocaptions|>");
    int no_timestamps_token = tokenizer_convert_token_to_id(tokenizer, "<|notimestamps|>");
    int start_of_prev_token = tokenizer_convert_token_to_id(tokenizer, "<|startofprev|>");
    int special_token_begin = tokenizer_convert_token_to_id(tokenizer, "<|endoftext|>");

    SpecialTokens special_tokens{
        sot_token,       eot_token,        blank_token,     no_timestamps_token, timestamp_begin_token,
        no_speech_token, transcribe_token, translate_token, special_token_begin};
    tokenizer->specialTokens = special_tokens;
}

void init_non_speech_tokens(Tokenizer *tokenizer) {
    std::vector<std::string> non_speech_tokens{"!",
                                               "\"",
                                               "#",
                                               "(",
                                               ")",
                                               "*",
                                               "+",
                                               "/",
                                               ":",
                                               ";",
                                               "<",
                                               "=",
                                               ">",
                                               "@",
                                               "[",
                                               "\\",
                                               "]",
                                               "^",
                                               "_",
                                               "`",
                                               "{",
                                               "|",
                                               "}",
                                               "~",
                                               " (",
                                               " \"",
                                               "--",
                                               " -",
                                               " [",
                                               " '",
                                               " =",
                                               " |",
                                               " :",
                                               " /",
                                               " )",
                                               " <",
                                               " #",
                                               " +",
                                               " --",
                                               " {",
                                               " *",
                                               " }",
                                               " >",
                                               " ;",
                                               " ]",
                                               " @",
                                               " \\",
                                               "))",
                                               ">>",
                                               " `",
                                               " _",
                                               " ~",
                                               " (\"",
                                               "---",
                                               "(\"",
                                               " >>",
                                               " <<",
                                               " ^",
                                               "('",
                                               " ---",
                                               "}}",
                                               "]]",
                                               " >>>",
                                               "「",
                                               "」",
                                               " ((",
                                               " ))",
                                               " [[",
                                               "<<",
                                               "�",
                                               " (\'",
                                               "((",
                                               " �",
                                               ")))",
                                               " {{",
                                               "{{",
                                               "[[",
                                               "-(",
                                               ">>>",
                                               " }}",
                                               " 「",
                                               "『",
                                               "』",
                                               " )))",
                                               "-[",
                                               "<|startoftranscript|>",
                                               "<|translate|>",
                                               "<|transcribe|>",
                                               "<|startoflm|>",
                                               "<|startofprev|>",
                                               "<|nocaptions|>"};
    tokenizer->numNonSpeechTokens = non_speech_tokens.size();
    tokenizer->nonSpeechTokens = (int *)malloc(sizeof(int) * tokenizer->numNonSpeechTokens);
    for (auto i = 0; i < tokenizer->numNonSpeechTokens; i++) {
        tokenizer->nonSpeechTokens[i] = tokenizer_convert_token_to_id(tokenizer, non_speech_tokens[i].c_str());
    }
}

int tokenizer_convert_token_to_id(const Tokenizer *tokenizer, const char *token_string) {
    // Encode token
    CEncoding *encoding = tokenizer_encode(tokenizer->handle, token_string, false);
    // Get length of encoding
    size_t length = encoding_get_length(encoding);
    // Ensure length is 1 so only a single token was extracted
    assert(length == 1);
    // Get token id
    uint32_t id = encoding_get_ids(encoding, &length)[0];

    free_encoding_array(encoding, length);

    return static_cast<int>(id);
}

Tokenizer *tokenizer_init_from_file(const char *path) {
    // Dynamically allocate tokenizer memory
    Tokenizer *tokenizer = (Tokenizer *)malloc(sizeof(Tokenizer));
    if (!tokenizer) {
        LOGE("Out of memory error while initializing tokenizer!");
        return NULL;
    }

    // Load file to check existence and get vocabulary size.
    std::ifstream file(path);
    if (!file) {
        LOGE("Error loading provided tokenizer JSON. File may not exist!\n");
        return NULL;
    }

    // Get and set vocabulary size
    auto json_file = std::make_unique<json>(json::parse(file));
    if (!json_file) {
        LOGE("Error parsing the provided tokenizer JSON!");
        return NULL;
    }
    tokenizer->vocabSize = (*json_file)["model"]["vocab"].size();
    LOGI("postproc vocab size: %d\n", tokenizer->vocabSize);

    // Load vocabulary from tokenizer file
    tokenizer->handle = tokenizer_from_file(path);
    if (!tokenizer->handle) {
        LOGE("Error unable to initialize tokenizer from provided file!");
        return NULL;
    }

    init_special_tokens(tokenizer);
    init_non_speech_tokens(tokenizer);
    return tokenizer;
}

char *tokenizer_decode(const Tokenizer *tokenizer, const int *tokens, int tokenCount, bool skipSpecialTokens) {
    return tokenizer_decode(tokenizer->handle, reinterpret_cast<const unsigned int *>(tokens), tokenCount,
                            skipSpecialTokens);
}

void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer) {
        if (tokenizer->handle) {
            tokenizer_free(tokenizer->handle);
            tokenizer->handle = nullptr;
        }
        if (tokenizer->nonSpeechTokens) {
            free((void *)tokenizer->nonSpeechTokens);
            tokenizer->nonSpeechTokens = nullptr;
        }
        free((void *)tokenizer);
    }
}

void tokenizer_free_rstring(char *s) { free_rstring(s); }
