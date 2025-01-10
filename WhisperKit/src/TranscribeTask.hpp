#pragma once

#include "WhisperKitConfiguration.hpp"
#include "WhisperKitTranscriptionResult.hpp"
#include "nlohmann/json.hpp"
#include <thread>
#include <string>
#include <unistd.h>
#include <memory>

namespace WhisperKit::TranscribeTask {
class AudioCodec;
class Runtime;
}

struct TranscribeTask {
    std::string audio_file;
    std::string model_size;
    float duration;

    void transcribe(const char* audio_file, whisperkit_transcription_result_t* transcription_result);
    TranscribeTask(const whisperkit_configuration_t& config);
    ~TranscribeTask();
    private: 
        void textOutputProc();
        whisperkit_configuration_t config;
        std::unique_ptr<nlohmann::json> argsjson;
        std::unique_ptr<std::thread> text_out_thread;
        std::unique_ptr<WhisperKit::TranscribeTask::AudioCodec> audio_codec;
        std::unique_ptr<WhisperKit::TranscribeTask::Runtime> runtime;

};



