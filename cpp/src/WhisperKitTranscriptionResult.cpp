
#include "WhisperKitTranscriptionResult.hpp"

whisperkit_transcription_result_t::whisperkit_transcription_result_t() {
    curr_transcription.clear();
    all_transcription.clear();
}

whisperkit_transcription_result_t::~whisperkit_transcription_result_t() {
    curr_transcription.clear();
    all_transcription.clear();
}

void whisperkit_transcription_result_t::set_transcription(const std::string& input_transcription) {
    curr_transcription = std::move(input_transcription);
    all_transcription.append(curr_transcription);
}

std::string whisperkit_transcription_result_t::get_chunk_transcription() const { return curr_transcription; }

std::string whisperkit_transcription_result_t::get_transcription() const { return all_transcription; }