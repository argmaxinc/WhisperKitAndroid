//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <tflite_model.hpp>
#include <vector>

using namespace std;

class AudioChunk {
   public:
    int start_index;
    int num_samples;
    float* data;
};

class AudioInputModel : public TFLiteModel {
   public:
    AudioInputModel(const string input_file);
    virtual ~AudioInputModel() {};

    bool initialize(string model_path, string lib_dir, string cache_dir, int backend = kUndefinedBackend);
    void uninitialize();
    virtual void invoke(bool measure_time = false);

    float get_next_chunk(char* output);

   private:
    uint32_t _sample_size;
    float _curr_timestamp = 0.0;
    string _audio_input_file;
    const float _energy_threshold = 0.02;
    const int _frame_length_samples = (0.1 * 16000);

    vector<float> _sample_buffer;
    vector<AudioChunk> _chunks;

    int move_and_fill_pcm(char* output);
    void read_audio_file(string input_file);
    void chunk_all();
    uint32_t split_on_middle_silence(uint32_t start_index, uint32_t end_index);
};
