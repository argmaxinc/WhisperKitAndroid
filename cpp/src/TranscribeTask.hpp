#pragma once

#include <unistd.h>

#include <memory>
#include <string>
#include <thread>

#include "WhisperKitConfiguration.hpp"
#include "WhisperKitTranscriptionResult.hpp"
#include "nlohmann/json.hpp"

namespace WhisperKit::TranscribeTask {
class AudioCodec;
class Runtime;
}  // namespace WhisperKit::TranscribeTask

struct TranscribeTask {
    std::string audio_file;
    std::string model_size;
    whisperkit_transcription_result_t* _transcription;
    float duration;

    // audio file transcription
    void transcribe(const char* audio_file, whisperkit_transcription_result_t* transcription_result);
    // audio stream mode: init, append, close
    void initStreaming(whisperkit_transcription_result_t* transcription_result, int sample_rate = 0,
                       int num_channels = 0);
    bool appendAudio(int size, char* buffer0, char* buffer1 = nullptr);
    void closeStreaming();

    TranscribeTask(const whisperkit_configuration_t& config);
    ~TranscribeTask();

   private:
    void textOutputProc();
    int chunk_idx;
    whisperkit_configuration_t config;
    std::unique_ptr<nlohmann::json> argsjson;
    std::unique_ptr<std::thread> text_out_thread;
    std::unique_ptr<WhisperKit::TranscribeTask::AudioCodec> audio_codec;
    std::unique_ptr<WhisperKit::TranscribeTask::Runtime> runtime;
};
