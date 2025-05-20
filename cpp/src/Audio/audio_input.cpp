//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#include "audio_input.hpp"

// 30 seconds of PCM audio samples
constexpr const int MAX_CHUNK_LENGTH = (16000 * 30);
constexpr const int INTERNAL_AUDIO_SIZE = (1.5 * MAX_CHUNK_LENGTH);

using namespace std;

AudioBuffer::AudioBuffer() {
    _swr = nullptr;
    _source_frame = nullptr;
    _target_frame = nullptr;

    _tgt_bytes_per_sample = 0;
    _src_bytes_per_sample = 0;
    _verbose = false;
}

AudioBuffer::~AudioBuffer() { uninitialize(); }

bool AudioBuffer::initialize(AVFrame* src_frame, AVFrame* tgt_frame, bool verbose) {
    lock_guard<mutex> lock(_mutex);

    _verbose = verbose;
    _source_frame = src_frame;
    _target_frame = tgt_frame;
    _src_bytes_per_sample = av_get_bytes_per_sample((AVSampleFormat)_source_frame->format);
    if (_source_frame->format < AV_SAMPLE_FMT_U8P && _source_frame->ch_layout.nb_channels >= 2) {
        // interleaved, stereo format
        _src_bytes_per_sample *= _source_frame->ch_layout.nb_channels;
    }
    _tgt_bytes_per_sample = av_get_bytes_per_sample((AVSampleFormat)_target_frame->format);

    _swr = swr_alloc();
    av_opt_set_chlayout(_swr, "in_chlayout", &_source_frame->ch_layout, 0);
    av_opt_set_int(_swr, "in_sample_rate", _source_frame->sample_rate, 0);
    av_opt_set_sample_fmt(_swr, "in_sample_fmt", (AVSampleFormat)_source_frame->format, 0);

    av_opt_set_chlayout(_swr, "out_chlayout", &_target_frame->ch_layout, 0);
    av_opt_set_int(_swr, "out_sample_rate", _target_frame->sample_rate, 0);
    av_opt_set_sample_fmt(_swr, "out_sample_fmt", (AVSampleFormat)_target_frame->format, 0);
    int ret = swr_init(_swr);
    if (ret < 0) {
        LOGE("Error in swr_init: %s\n", *av_err2string(ret));
        return false;
    }
    return true;
}

void AudioBuffer::uninitialize() {
    lock_guard<mutex> lock(_mutex);

    if (_swr) {
        swr_free(&_swr);
        _swr = nullptr;
    }

    _buffer.clear();

    _source_frame = nullptr;
    _target_frame = nullptr;
}

int AudioBuffer::append(int bytes, char* input0, char* input1) {
    lock_guard<mutex> lock(_mutex);

    av_frame_unref(_target_frame);
    _target_frame->ch_layout = AV_CHANNEL_LAYOUT_MONO;
    _target_frame->sample_rate = SAMPLE_FREQ;
    _target_frame->format = AV_SAMPLE_FMT_FLT;

    if (_source_frame->sample_rate == SAMPLE_FREQ && _source_frame->ch_layout.nb_channels == 1 &&
        _source_frame->format == AV_SAMPLE_FMT_FLT) {
        _buffer.insert(_buffer.end(), reinterpret_cast<float*>(input0),
                       reinterpret_cast<float*>(input0) + (bytes / sizeof(float)));

        _target_frame->nb_samples = bytes / _tgt_bytes_per_sample;
    } else {
        _source_frame->data[0] = (uint8_t*)input0;
        if (input1 != nullptr)  // for planar, ch > 1 audio frame source
            _source_frame->data[1] = (uint8_t*)input1;
        _source_frame->nb_samples = bytes / _src_bytes_per_sample;

        int ret = swr_convert_frame(_swr, _target_frame, _source_frame);
        if (ret < 0) {
            LOGE("Error in swr_convert_frame: %s\n", *av_err2string(ret));
            return -1;
        }

        _buffer.insert(_buffer.end(), reinterpret_cast<float*>(_target_frame->extended_data[0]),
                       reinterpret_cast<float*>(_target_frame->extended_data[0]) + (_target_frame->nb_samples));
    }

    return _target_frame->nb_samples;
}

void AudioBuffer::print_frame_info() {
    if (_verbose) {
        LOGI("source rate: %d, ch: %d, format: %d, samples: %d\n", _source_frame->sample_rate,
             _source_frame->ch_layout.nb_channels, (int)_source_frame->format, _source_frame->nb_samples);
        LOGI("target rate: %d, ch: %d, format: %d, samples: %d\n", _target_frame->sample_rate,
             _target_frame->ch_layout.nb_channels, (int)_target_frame->format, _target_frame->nb_samples);
    }
}

int AudioBuffer::samples(int desired_samples) {
    if (desired_samples == 0) {
        return _buffer.size();
    } else if (_buffer.size() > desired_samples) {
        return desired_samples;
    } else
        return _buffer.size();
}

void AudioBuffer::consumed(int samples) {
    lock_guard<mutex> lock(_mutex);

    if (_buffer.size() >= samples) {
        _buffer.erase(_buffer.begin(), _buffer.begin() + samples);
    } else {
        LOGE("requested samples (%d) > available (%ld)\n", samples, _buffer.size());
        _buffer.clear();
    }
}

// AudioInputModel
AudioInputModel::AudioInputModel(  // buffer input mode
    int freq, int channels, int format) {
    _source_frame = av_frame_alloc();
    _target_frame = av_frame_alloc();

    _source_frame->sample_rate = freq;
    if (channels == 1)
        _source_frame->ch_layout = AV_CHANNEL_LAYOUT_MONO;
    else if (channels == 2)
        _source_frame->ch_layout = AV_CHANNEL_LAYOUT_STEREO;
    else
        throw std::invalid_argument("more than 2 audio channels not supported");

    if (format <= AV_SAMPLE_FMT_NONE)
        _source_frame->format = AV_SAMPLE_FMT_S16;
    else
        _source_frame->format = format;
    _target_frame->ch_layout = AV_CHANNEL_LAYOUT_MONO;
    _target_frame->sample_rate = SAMPLE_FREQ;
    _target_frame->format = AV_SAMPLE_FMT_FLT;

    _model = std::make_unique<TFLiteModel>("audio_input");

    _pcm_buffer = make_unique<AudioBuffer>();
}

AudioInputModel::~AudioInputModel() {
    av_frame_unref(_source_frame);
    av_frame_free(&_source_frame);
    _source_frame = nullptr;
    av_frame_unref(_target_frame);
    av_frame_free(&_target_frame);
    _target_frame = nullptr;
}

bool AudioInputModel::initialize(bool debug) {
    if (!_model->initializeModelInMemory(WhisperKit::InMemoryModel::ModelType::kSimpleVADModel, debug)) {
        LOGE("Failed to initialize\n");
        return false;
    }

    if (!_pcm_buffer->initialize(_source_frame, _target_frame, debug)) {
        LOGE("Failed to initialize PCM buffer class\n");
        return false;
    }

    _pcm_buffer->print_frame_info();

    return true;
}

void AudioInputModel::uninitialize() {
    _float_buffer.clear();
    _pcm_buffer->uninitialize();

    _model->uninitialize();
}

void AudioInputModel::invoke(bool measure_time) { _model->invoke(measure_time); }

float AudioInputModel::get_next_chunk(char* output) {
    int audio_bytes = 0, audio_samples = 0;
    int ret, i;

    memset(output, 0, MAX_CHUNK_LENGTH * sizeof(float));
    if (_pcm_buffer->samples() == 0) {
        return -1.0;
    }

    if (_remain_samples < MAX_CHUNK_LENGTH) {
        audio_samples = get_next_samples();
        if (audio_samples < MAX_CHUNK_LENGTH / 300) {  // less than 0.1s
            return -1.0;
        }
    }
    chrono::time_point<chrono::high_resolution_clock> before_exec = chrono::high_resolution_clock::now();

    auto start_time = get_silence_index(output, audio_samples);

    auto& latencies = _model->_latencies;
    auto after_exec = chrono::high_resolution_clock::now();
    float interval_infs = chrono::duration_cast<std::chrono::microseconds>(after_exec - before_exec).count() / 1000.0;
    latencies.push_back(interval_infs);

    return start_time;
}

float AudioInputModel::get_silence_index(char* output, int audio_samples) {
    uint32_t max_index, end_index;
    max_index = _silence_index + _remain_samples + audio_samples;
    if (_silence_index + MAX_CHUNK_LENGTH <= max_index) {
        end_index = split_on_middle_silence(max_index);
    } else {
        end_index = max_index;
    }
    if (end_index <= _silence_index) {
        return -1.0;
    }

    _remain_samples = max_index - end_index;
    auto start_time = (float)_silence_index / SAMPLE_FREQ;
    memcpy(output, _float_buffer.data(), (end_index - _silence_index) * sizeof(float));

    _float_buffer.erase(_float_buffer.begin(), _float_buffer.begin() + (end_index - _silence_index));

    _silence_index = end_index;

    return start_time;
}

uint32_t AudioInputModel::split_on_middle_silence(uint32_t max_index) {
    auto mid_index = _silence_index + (max_index - _silence_index) / 2;

    auto count = (int)ceil((float)(max_index - mid_index) / _frame_length_samples);
    vector<bool> voices;

    // calculateVoiceActivityInChunks
    auto inputs = _model->get_input_ptrs();
    memcpy(inputs[0].first, (char*)&_float_buffer[mid_index - _silence_index],
           (count * _frame_length_samples) * sizeof(float));
    memcpy(inputs[1].first, &_energy_threshold, sizeof(float));

    _model->invoke();

    auto outputs = _model->get_output_ptrs();

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

std::vector<std::pair<char*, int>> AudioInputModel::get_input_ptrs() {
    if (_model == nullptr) {
        return {};
    }
    return _model->get_input_ptrs();
}

std::vector<std::pair<char*, int>> AudioInputModel::get_output_ptrs() {
    if (_model == nullptr) {
        return {};
    }
    return _model->get_output_ptrs();
}

void AudioInputModel::fill_pcmdata(int bytes, char* pcm_buffer0, char* pcm_buffer1) {
    int ret = _pcm_buffer->append(bytes, pcm_buffer0, pcm_buffer1);

    _curr_buf_time = (_pcm_buffer->samples() + _remain_samples) / _target_frame->sample_rate;
    _total_src_bytes += bytes;
}

float AudioInputModel::get_total_input_time() {
    return (_total_src_bytes / (_source_frame->sample_rate * _pcm_buffer->get_srcbytes_per_sample()));
}

int AudioInputModel::get_next_samples() {
    auto remaining_time_x100 = (unsigned int)(_remain_samples * 100) / SAMPLE_FREQ;

    int max_target_samples = (int)(_target_frame->sample_rate * (3000 - remaining_time_x100) / 100);
    auto target_samples = _pcm_buffer->samples(max_target_samples);
    if (target_samples <= 0) {
        LOGE(" audio buffer size is 0\n");
        return -1;
    }

    auto audio_buffer = _pcm_buffer->get_buffer();
    _float_buffer.insert(_float_buffer.end(), audio_buffer, audio_buffer + target_samples);
    _pcm_buffer->consumed(target_samples);
    return target_samples;
}
