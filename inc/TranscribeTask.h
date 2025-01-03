#pragma once

#include "WhisperKitConfiguration.h"
#include "nlohmann/json.hpp"
#include <thread>
#include <string>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>
#include <memory>

namespace WhisperKit::TranscribeTask {
class AudioCodec;
}

struct TranscribeTask {
    std::string audio_file;
    std::string model_size;
    float duration;

    void transcribe(const char* audio_file, char** transcription);
    TranscribeTask(const whisperkit_configuration_t& config);
    ~TranscribeTask();
    private: 
        void textOutputProc();
        whisperkit_configuration_t config;
        std::unique_ptr<nlohmann::json> argsjson;
        std::unique_ptr<std::thread> text_out_thread;
        std::unique_ptr<WhisperKit::TranscribeTask::AudioCodec> audio_codec;

};



