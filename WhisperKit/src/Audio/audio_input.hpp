//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

extern "C" {
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

#include "backend_class.hpp"

constexpr const int SAMPLE_FREQ = 16000;

inline std::unique_ptr<char*> av_err2string(int errnum){
    return std::make_unique<char*>(av_make_error_string( 
        (char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), 
        AV_ERROR_MAX_STRING_SIZE, errnum 
    ));
} 

class AudioBuffer {
private: 
    SwrContext *_swr; 
    AVFrame *_source_frame; 
    AVFrame *_target_frame; 
    std::mutex _mutex;
    bool _verbose; 

    // target buf associated, with 16khz, mono PCM data
    uint32_t _end_index; // unit of short int
    std::vector<float> _buffer; 
    int _tgt_bytes_per_sample;
    int _src_bytes_per_sample;

public:
    AudioBuffer();
    ~AudioBuffer();

    bool initialize(AVFrame *src_frame, AVFrame *tgt_frame, bool verbose = false);
    void uninitialize();
    bool empty_source();

    int append(int bytes, char* buffer0, char* buffer1 = nullptr);
    int samples(int desired_samples = 0);
    void consumed(int samples);
    float* get_buffer()     { return _buffer.data(); }
    int get_srcbytes_per_sample()   { return _src_bytes_per_sample; }
    void print_frame_info();
};

class AudioInputModel: public MODEL_SUPER_CLASS {
public:
    AudioInputModel(int freq, int channels, int format = AV_SAMPLE_FMT_FLT);
    ~AudioInputModel();

    bool initialize(
        std::string model_path, 
        std::string lib_dir,
        std::string cache_dir, 
        int backend, 
        bool debug=false);
    void uninitialize();
    virtual void invoke(bool measure_time=false);

    void fill_pcmdata(
        int size, 
        char* pcm_buffer0, 
        char* pcm_buffer1=nullptr
    );
    float get_next_chunk(char* output);
    int get_curr_buf_time() { return _curr_buf_time; }
    float get_total_input_time();
    bool empty_source() { return _pcm_buffer->empty_source(); }

private:
    int32_t _total_src_bytes = 0;
    int32_t _buffer_index = 0;

    std::unique_ptr<AudioBuffer> _pcm_buffer;

    AVFrame *_source_frame; 
    AVFrame *_target_frame; 

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
