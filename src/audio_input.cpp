//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#include <audio_input.hpp>

#define MAX_CHUNK_LENGTH    (16000 * 30) // 30 seconds of PCM audio samples

using namespace std;

AudioBuffer::AudioBuffer() 
{
    _buffer = nullptr;
    _source_spec = nullptr;
    _stream = nullptr;
    _cap_bytes = 0;
    _bytes_per_sample = sizeof(short int);
    _end_index = 0;
}

AudioBuffer::~AudioBuffer() {
    uninitialize();
}

void AudioBuffer::initialize(
    SDL_AudioSpec* src_spec,
    SDL_AudioSpec* tgt_spec
){
    lock_guard<mutex> lock(_mutex);

    _source_spec = src_spec;
    _target_spec = tgt_spec;
    if(_source_spec->format == SDL_AUDIO_F32){
        _bytes_per_sample = sizeof(float);
    } else {
        _bytes_per_sample = sizeof(short int);
    }
    _cap_bytes = (int)(1.5 * MAX_CHUNK_LENGTH * _bytes_per_sample);
    _buffer = (short int*)SDL_malloc(_cap_bytes);

    _stream = SDL_CreateAudioStream(_source_spec, _target_spec);
    if (_stream == nullptr) {
        LOGE("Failed to create audio stream: %s\n", SDL_GetError());
        return;
    }

    LOGI("Stream: freq - %d, channels - %d, format - %d, target_buf size - %d\n", 
        _source_spec->freq, 
        _source_spec->channels, 
        _source_spec->format,
        _cap_bytes
    );
}

void AudioBuffer::uninitialize(){
    lock_guard<mutex> lock(_mutex);

    if (_stream == nullptr)
        return;

    SDL_free(_buffer);
    SDL_DestroyAudioStream(_stream);
    SDL_Quit();

    _buffer = nullptr;
    _source_spec = nullptr;
    _stream = nullptr;
    _cap_bytes = 0;
}

int AudioBuffer::append(int bytes, char* input) {
    lock_guard<mutex> lock(_mutex);

    bool bRet = false;

        bRet = SDL_PutAudioStreamData(
            _stream, input, bytes
        );
        if (!bRet) {
            LOGE("Failed from SDL_PutAudioStreamData: %s\n", SDL_GetError());
            return 0;
        }
        SDL_FlushAudioStream(_stream);

        auto target_bytes = SDL_GetAudioStreamAvailable(_stream);
        if ((_end_index * _bytes_per_sample + target_bytes) > _cap_bytes) {
            LOGE("buffer overflow..\n");
            return 0;
        }

        auto ret = SDL_GetAudioStreamData(
        _stream, &_buffer[_end_index], target_bytes
        );
        if (ret < 0) {
            LOGE("Failed from SDL_GetAudioStreamData: %s\n", SDL_GetError());
            return 0;
        }
    auto samples = (target_bytes / _bytes_per_sample);
    _end_index += samples;

    return samples;
}

int AudioBuffer::samples(int desired_samples) {
    if (desired_samples == 0) {
        return _end_index;
    } else if (_end_index > desired_samples){
        return desired_samples;
    } else 
        return _end_index;
}

void AudioBuffer::consumed(int samples){
    lock_guard<mutex> lock(_mutex);

    if(_end_index > samples){
        // move the remaining samples to the beginning of _target_buffer
        memmove(
            _buffer, 
            &_buffer[_end_index], 
            (_end_index - samples) * _bytes_per_sample
        );
        _end_index -= samples;
    } else {
        if(_end_index < samples){
            LOGE("requested samples (%d) > available (%d)\n", 
                samples, _end_index
            );
        }
        _end_index = 0;
    }
}

// AudioInputModel
AudioInputModel::AudioInputModel( // buffer input mode
   int freq, int channels, int format
) :MODEL_SUPER_CLASS("audio_input")
{
    _source_spec.freq = freq;
    _source_spec.channels = channels;
    _target_spec.channels = 1;
    _target_spec.freq = SAMPLE_FREQ;

    if (format == SAMPLE_FMT_FLT) {
        _source_spec.format = SDL_AUDIO_F32;
        _target_spec.format = SDL_AUDIO_F32;
    } else {
        _source_spec.format = SDL_AUDIO_S16;
        _target_spec.format = SDL_AUDIO_S16;
    }

    _pcm_buffer = make_unique<AudioBuffer>();
}

bool AudioInputModel::initialize(
    string model_file,
    string lib_dir,
    string cache_dir, 
    int backend, 
    bool debug
){
    SDL_SetMainReady();
    if (!SDL_Init(0)) {
        LOGE("Couldn't initialize SDL: %s", SDL_GetError());
        return false;
    }

    if(!MODEL_SUPER_CLASS::initialize(model_file, lib_dir, cache_dir, backend, debug)){
        LOGE("Failed to initialize\n");
        return false;
    }

    _float_buffer.resize(1.5 * MAX_CHUNK_LENGTH);
    _pcm_buffer->initialize(&_source_spec, &_target_spec);

    return true;
}

void AudioInputModel::uninitialize() {
    _float_buffer.clear();
    _pcm_buffer->uninitialize();

    MODEL_SUPER_CLASS::uninitialize();
}

void AudioInputModel::invoke(bool measure_time){
    MODEL_SUPER_CLASS::invoke(measure_time);
}

float AudioInputModel::get_next_chunk(char* output){
    int audio_bytes = 0, audio_samples = 0;
    int ret, i;

    memset(output, 0, MAX_CHUNK_LENGTH * sizeof(float));
    if (_pcm_buffer->samples() == 0){
        return -1.0;
    }

    chrono::time_point<chrono::high_resolution_clock> 
        before_exec = chrono::high_resolution_clock::now();
    if (_remain_samples < MAX_CHUNK_LENGTH) {
        audio_samples = get_next_samples();
        if (audio_samples < 0) {
            return audio_samples;
        }
    }

    auto start_time = get_silence_index(output, audio_samples);

    auto after_exec = chrono::high_resolution_clock::now();
    float interval_infs =
        chrono::duration_cast<std::chrono::microseconds>(
            after_exec - before_exec).count() / 1000.0;
    _latencies.push_back(interval_infs);  

    return start_time;
}

float AudioInputModel::get_silence_index(char* output, int audio_samples){
    uint32_t max_index, end_index;
    max_index = _silence_index + _remain_samples + audio_samples;
    if (_silence_index + MAX_CHUNK_LENGTH <= max_index) {
        end_index = split_on_middle_silence(max_index);
    }
    else {
        end_index = max_index;
    }
    if (end_index <= _silence_index) {
        return -1.0;
    }

    _remain_samples = max_index - end_index;
    auto start_time = (float)_silence_index / SAMPLE_FREQ;
    memcpy(output, &_float_buffer[0], (end_index - _silence_index) * sizeof(float));

    // move the remaining samples to the beginning of _float_buffer
    memmove(
        &_float_buffer[0], 
        &_float_buffer[end_index - _silence_index], 
        _remain_samples * sizeof(float)
    );

    _silence_index = end_index;    

    return start_time;
}

uint32_t AudioInputModel::split_on_middle_silence(uint32_t max_index)
{
    auto mid_index = _silence_index + (max_index - _silence_index) / 2;

    auto count = (int)ceil((float)(max_index - mid_index) / _frame_length_samples);
    vector<bool> voices;

    // calculateVoiceActivityInChunks
    auto inputs = get_input_ptrs();
    memcpy(
        inputs[0].first, 
        (char*)&_float_buffer[mid_index - _silence_index], 
        (count * _frame_length_samples) * sizeof(float)
    );
    memcpy(inputs[1].first, &_energy_threshold, sizeof(float));

    invoke();

    auto outputs = get_output_ptrs();

    // find the Longest Silence 
    // all indices here mean voices index from above, not audio samples
    int longest_count = 0, longest_start = 0, longest_end = 0;
    int idx = 0, is_voice = 0;
    auto output_size = outputs[0].second / sizeof(float);
    auto output = reinterpret_cast<float*>(outputs[0].first);

    while (idx < output_size) {
        if (output[idx] > 0) {
            idx++;
            is_voice++;
            continue;
        }
        // silence starts at idx
        auto endidx = idx;
        while (endidx < output_size && output[endidx] <= 0) {
            endidx++;
        }

        auto count = endidx - idx;
        if (count > longest_count) {
            longest_count = count;
            longest_start = idx;
            longest_end = endidx;
        }
        idx = endidx;
    }

    if (longest_count == 0) {
        return max_index;
    }

    auto silence_mid_idx = longest_start + (longest_end - longest_start) / 2;
    // voice activity index to audio sample index + mid_index
    return mid_index + silence_mid_idx * _frame_length_samples;
}

void AudioInputModel::fill_pcmdata(int bytes, char* pcm_buffer){
    int ret = _pcm_buffer->append(bytes, pcm_buffer);

    _curr_buf_time = (_pcm_buffer->samples() + _remain_samples) 
                    / _target_spec.freq;
    _total_src_bytes += bytes; 
}

int AudioInputModel::get_next_samples()
{
    auto remaining_time_x100 = 
        (unsigned int)(_remain_samples * 100)/ SAMPLE_FREQ;

    int max_target_samples = 
        (int)(_target_spec.freq * (3000 - remaining_time_x100)/100);
    auto target_samples = _pcm_buffer->samples(max_target_samples);
    if (target_samples <= 0) {
        LOGE(" audio buffer is null\n");
        return -1;
    }

    auto audio_buffer = _pcm_buffer->get_buffer();
    if (_source_spec.format == SDL_AUDIO_F32){
        // it is already in float type format audio samples
        memcpy(
            &_float_buffer[_remain_samples], 
            audio_buffer, 
            target_samples * sizeof(float)
        );
    } else {
        // float type sample = short int type PCM / 32768.0
        for (int i = 0; i < target_samples; i++) {
            _float_buffer[_remain_samples + i] = 
                (float)audio_buffer[i] / 32768.0;
        }
    }
    _pcm_buffer->consumed(target_samples);
    return target_samples;
}