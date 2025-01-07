
#include "WhisperKitTranscriptionResult.hpp"

whisperkit_transcription_result_t::whisperkit_transcription_result_t() {

}

whisperkit_transcription_result_t::~whisperkit_transcription_result_t() {

}

void whisperkit_transcription_result_t::set_transcription(const std::string& input_transcription) {
    transcription = input_transcription;
}

std::string whisperkit_transcription_result_t::get_transcription() const {
    return transcription;
}