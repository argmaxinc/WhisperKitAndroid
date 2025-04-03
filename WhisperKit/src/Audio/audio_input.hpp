//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_audio.h>

#include "backend_class.hpp"

constexpr const int SAMPLE_FREQ = 16000;
constexpr const int SAMPLE_FMT_DEF = 0;     // AV_SAMPLE_FMT_NONE in ffmpeg
constexpr const int SAMPLE_FMT_FLT = 3;     // AV_SAMPLE_FMT_FLT in ffmpeg
constexpr const int SAMPLE_FMT_S16P = 6;    // AV_SAMPLE_FMT_S16P in ffmpeg
constexpr const int SAMPLE_FMT_FLTP = 8;    // AV_SAMPLE_FMT_FLTP in ffmpeg

class AudioBuffer {
private: 
    SDL_AudioSpec* _source_spec;
    SDL_AudioSpec* _target_spec;
    SDL_AudioStream* _stream;
    std::mutex _mutex;

    // target buf associated, with 16khz, mono PCM data
    uint32_t _end_index; // unit of short int
    uint32_t _cap_bytes;
    short int *_buffer;
    int _bytes_per_sample;

public:
    AudioBuffer();
    ~AudioBuffer();

    void initialize(
        SDL_AudioSpec* src_spec,
        SDL_AudioSpec* tgt_spec
    );
    void uninitialize();
    bool empty_source();

    int append(int bytes, char* buffer = nullptr);
    int samples(int desired_samples = 0);
    void consumed(int samples);
    short int* get_buffer() { return _buffer; }
};

class AudioInputModel {
public:
    AudioInputModel(int freq, int channels, int format = SAMPLE_FMT_DEF);
    virtual ~AudioInputModel() {};

    bool initialize(
        bool debug=false);
    void uninitialize();
    virtual void invoke(bool measure_time=false);

    // this is temporary
    std::vector<std::pair<char*, int>> get_input_ptrs();
    std::vector<std::pair<char*, int>> get_output_ptrs();

    void fill_pcmdata(int size, char* pcm_buffer=nullptr);
    float get_next_chunk(char* output);
    int get_curr_buf_time() { return _curr_buf_time; }
    float get_total_input_time() { 
        return (_total_src_bytes / (_source_spec.freq * _source_spec.channels * 2)); 
    }
    bool empty_source() { return _pcm_buffer->empty_source(); }

private:
    SDL_AudioStream* _stream = nullptr;

    std::unique_ptr<TFLiteModel> _model;

    int32_t _total_src_bytes = 0;
    int32_t _buffer_index = 0;

    std::unique_ptr<AudioBuffer> _pcm_buffer;

    SDL_AudioSpec _source_spec;
    SDL_AudioSpec _target_spec;

    const float _energy_threshold = 0.02;
    const int _frame_length_samples = (0.1 * 16000);

    std::vector<float> _float_buffer;
    int32_t _silence_index = 0;
    int32_t _remain_samples = 0;
    int _curr_buf_time = 0;

    void read_audio_file(std::string input_file);
    void chunk_all(); 
    float get_silence_index(char* output, int audio_samples);
    int get_next_samples();
    uint32_t split_on_middle_silence(uint32_t end_index);
};
