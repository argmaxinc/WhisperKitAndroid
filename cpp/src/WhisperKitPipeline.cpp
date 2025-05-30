#include "WhisperKitPipeline.hpp"

#include "TranscribeTask.hpp"
whisperkit_pipeline_t::whisperkit_pipeline_t() { status = WHISPERKIT_PIPELINE_STATUS_INITIALIZED; }

whisperkit_pipeline_status_t whisperkit_pipeline_t::get_state() const { return status; }

void whisperkit_pipeline_t::set_state(whisperkit_pipeline_status_t status) { this->status = status; }

void whisperkit_pipeline_t::set_configuration(const whisperkit_configuration_t* configuration) {
    if (configuration == nullptr) {
        return;
    }
    this->configuration = whisperkit_configuration_t(*configuration);
    status = WHISPERKIT_PIPELINE_STATUS_CONFIGURED;
}

whisperkit_pipeline_t::~whisperkit_pipeline_t() { transcribe_task.reset(); }

void whisperkit_pipeline_t::build() { transcribe_task = std::make_unique<TranscribeTask>(this->configuration); }
void whisperkit_pipeline_t::transcribe(const char* audio_file,
                                       whisperkit_transcription_result_t* transcription_result) {
    transcribe_task->transcribe(audio_file, transcription_result);
}

void whisperkit_pipeline_t::init_streaming(whisperkit_transcription_result_t* transcription_result, int sample_rate,
                                           int num_channels) {
    transcribe_task->initStreaming(transcription_result, sample_rate, num_channels);
    status = WHISPERKIT_PIPELINE_STATUS_AUDIOINIT;
}

bool whisperkit_pipeline_t::append_audio(int size, char* buffer) {
    bool transcribed = transcribe_task->appendAudio(size, buffer);
    return transcribed;
}

void whisperkit_pipeline_t::close_streaming() { transcribe_task->closeStreaming(); }
