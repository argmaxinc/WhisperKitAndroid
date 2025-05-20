#pragma once

#include <memory>

#include "WhisperKit.h"
#include "WhisperKitConfiguration.hpp"

struct TranscribeTask;
struct whisperkit_transcription_result_t;
struct whisperkit_pipeline_t {
   public:
    whisperkit_pipeline_t();
    whisperkit_pipeline_t(const whisperkit_pipeline_t&) = delete;
    whisperkit_pipeline_t& operator=(const whisperkit_pipeline_t&) = delete;
    ~whisperkit_pipeline_t();
    whisperkit_pipeline_status_t get_state() const;
    void set_state(whisperkit_pipeline_status_t status);
    void set_configuration(const whisperkit_configuration_t* configuration);
    void build();
    // transcribe an audio file
    void transcribe(const char* audio_file, whisperkit_transcription_result_t* transcription_result);
    // in streaming mode: append any length of audio data
    void init_streaming(whisperkit_transcription_result_t* transcription_result, int sample_rate, int num_channels);
    bool append_audio(int size, char* buffer);
    void close_streaming();

   private:
    whisperkit_configuration_t configuration;
    whisperkit_pipeline_status_t status;
    std::unique_ptr<TranscribeTask> transcribe_task;
};