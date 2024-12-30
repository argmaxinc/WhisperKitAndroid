#include "WhisperKitPipeline.h"    

whisperkit_pipeline_t::whisperkit_pipeline_t() {
    status = WHISPERKIT_PIPELINE_STATUS_INITIALIZED;
}

whisperkit_pipeline_status_t whisperkit_pipeline_t::get_state() const { 
    return status; 
}

void whisperkit_pipeline_t::set_state(whisperkit_pipeline_status_t status) {
    this->status = status;
}

void whisperkit_pipeline_t::set_configuration(const whisperkit_configuration_t* configuration) {
    if(configuration == nullptr) {
        return;
    }
    this->configuration = whisperkit_configuration_t(*configuration);
    status = WHISPERKIT_PIPELINE_STATUS_CONFIGURED;
}

void whisperkit_pipeline_t::build() {


}
void whisperkit_pipeline_t::transcribe(const char* audio_file, char** transcription) {


}
