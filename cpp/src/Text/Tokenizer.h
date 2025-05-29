#ifndef TOKENIZER_API_H
#define TOKENIZER_API_H

#ifdef __cplusplus
extern "C" {
#endif
typedef struct TokenizerHandle TokenizerHandle;
#ifdef __cplusplus
}
#endif

// Tokenizer-related constants and structures
typedef struct {
    int startOfTranscriptToken;
    int endOfTranscriptToken;
    int blankToken;
    int noTimestampsToken;
    int timestampBeginToken;
    int noSpeechToken;
    int transcribeToken;
    int translateToken;
    int englishToken;
    int specialTokenBegin;
} SpecialTokens;

typedef struct {
    SpecialTokens specialTokens;
    int* nonSpeechTokens;
    int numNonSpeechTokens;
    unsigned int vocabSize;
    TokenizerHandle* handle;
} Tokenizer;

// Initialize the tokenizer
Tokenizer* tokenizer_init_from_file(const char* path, const char* config_path);

// Decode token IDs into a string
char* tokenizer_decode(const Tokenizer* tokenizer, const int* tokens, int tokenCount, bool skipSpecialTokens);

bool tokenizer_is_multilingual(const Tokenizer* tokenizer);

// Convert token string to ID
int tokenizer_convert_token_to_id(const Tokenizer* tokenizer, const char* tokenString);

// Get the size of the vocabulary
int tokenizer_get_vocab_size();

// Deallocates tokenizer
void tokenizer_free(Tokenizer* tokenizer);

// Deallocates decoded string
void tokenizer_free_rstring(char* s);

#endif  // TOKENIZER_API_H
