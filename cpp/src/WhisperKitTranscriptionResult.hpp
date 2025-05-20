#pragma once

#include <string>

struct whisperkit_transcription_result_t {
   public:
    whisperkit_transcription_result_t();
    ~whisperkit_transcription_result_t();

    void set_transcription(const std::string& transcription);
    std::string get_transcription() const;
    std::string get_chunk_transcription() const;

   private:
    std::string curr_transcription;
    std::string all_transcription;
};