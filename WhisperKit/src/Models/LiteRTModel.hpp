#pragma once

#include <string> 


struct TfLiteModel;
struct TfLiteInterpreterOptions;
struct TfLiteInterpreter;


class WhisperLiteRTModel {
public:
    WhisperLiteRTModel(const std::string& name);
    virtual ~WhisperLiteRTModel();

    protected:
        std::unique_ptr<TfLiteModel> _model;
        std::unique_ptr<TfLiteInterpreterOptions> _options;
        std::unique_ptr<TfLiteInterpreter> _interpreter;
};


class AudioEncoder : public WhisperLiteRTModel {
public:
    AudioEncoder(const std::string& name);
    virtual ~AudioEncoder();

    private:
    std::unique_ptr<WhisperLiteRTModel> model;
};

class TextDecoder : public WhisperLiteRTModel {
public:
    TextDecoder(const std::string& name);
    virtual ~TextDecoder();

    private:
    std::unique_ptr<WhisperLiteRTModel> model;
};

class PostProcessing : public WhisperLiteRTModel {
public:
    PostProcessing(const std::string& name);
    virtual ~PostProcessing();

    private:
    std::unique_ptr<WhisperLiteRTModel> model;
};