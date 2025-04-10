#pragma once

#include "WhisperKit.h"
#include "WhisperKitConfiguration.hpp"
#include <memory>

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
        void transcribe(const char* audio_file, whisperkit_transcription_result_t* transcription_result);
        private:
            whisperkit_configuration_t configuration;
            whisperkit_pipeline_status_t status;
            std::unique_ptr<TranscribeTask> transcribe_task;

};