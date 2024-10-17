//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#include <audio_input.hpp>

#define WAVE_HEADER_BYTES 78
#define MAX_CHUNK_LENGTH (16000 * 30)  // 30 seconds of PCM audio samples
#define WINDOW_PADDING 16000           // 1 second of silence padding

#define USE_MODEL 1

AudioInputModel::AudioInputModel(const string input_file) : TFLiteModel("audio_input") {
    _audio_input_file = input_file;
}

bool AudioInputModel::initialize(string model_file, string lib_dir, string cache_dir, int backend) {
    if (!TFLiteModel::initialize(model_file, lib_dir, cache_dir, backend)) {
        cerr << "Failed to initialize" << endl;
        return false;
    }

    chrono::time_point<chrono::high_resolution_clock> before_exec = chrono::high_resolution_clock::now();

    read_audio_file(_audio_input_file);
    chunk_all();
    // for (auto chunk : _chunks) {
    //     cout << "chunk: " << chunk.start_index << ", " << (float)chunk.start_index / 16000
    //         << " ~ " << (float)(chunk.start_index + chunk.num_samples) / 16000
    //         << endl;
    // }

    auto after_exec = chrono::high_resolution_clock::now();
    float interval_infs = chrono::duration_cast<std::chrono::microseconds>(after_exec - before_exec).count() / 1000.0;
    _latencies.push_back(interval_infs);
    return true;
}

void AudioInputModel::uninitialize() {
    _sample_buffer.clear();
    _chunks.clear();
    TFLiteModel::uninitialize();
}

void AudioInputModel::invoke(bool measure_time) { TFLiteModel::invoke(measure_time); }

float AudioInputModel::get_next_chunk(char* output) {
    memset(output, 0, MAX_CHUNK_LENGTH * sizeof(float));

    if (_chunks.empty()) {
        return -1.0;
    }

    auto start_time = (float)_chunks[0].start_index / 16000;
    _curr_timestamp = (float)(_chunks[0].start_index + _chunks[0].num_samples) / 16000;

    memcpy(output, _chunks[0].data, _chunks[0].num_samples * sizeof(float));

    _chunks.erase(_chunks.begin());
    return start_time;
}

void AudioInputModel::read_audio_file(string input_file) {
    ifstream audio_file;

    int skip_bytes = 0;
    if (input_file.find(".wav") != string::npos || input_file.find(".wave") != string::npos) {
        skip_bytes = WAVE_HEADER_BYTES;
    }

    audio_file.open(input_file, ios::binary | ios::ate);
    audio_file.seekg(0, audio_file.end);
    _sample_size = (uint32_t)((int)audio_file.tellg() - skip_bytes) / sizeof(short int);
    audio_file.seekg(skip_bytes, audio_file.beg);
    _sample_buffer.resize(_sample_size);

    // // middle of the buffer to the end: use it as a temp PCM buffer
    auto* short_int_buf = (short int*)&_sample_buffer[_sample_size / sizeof(short int)];
    audio_file.read(reinterpret_cast<char*>(short_int_buf), _sample_size * sizeof(short int));
    audio_file.close();

    int i;
    for (i = 0; i < _sample_size; i++) {
        // float type output = short int PCM input / 32768.0
        _sample_buffer[i] = (float)short_int_buf[i] / 32768.0;
    }
}

void AudioInputModel::chunk_all() {
    uint32_t start_index = 0, end_index = 0;

    // all indices here mean the audio sample index
    if (_sample_size < MAX_CHUNK_LENGTH) {
        AudioChunk chunk;
        chunk.start_index = start_index;
        chunk.num_samples = _sample_size;
        chunk.data = &_sample_buffer[0];
        _chunks.push_back(chunk);

        return;  // just a single chunk
    }

    while (start_index < _sample_size - WINDOW_PADDING) {
        end_index = _sample_size;
        if (start_index + MAX_CHUNK_LENGTH < end_index) {
            end_index = split_on_middle_silence(start_index, min(_sample_size, start_index + MAX_CHUNK_LENGTH));
        }

        if (end_index <= start_index) {
            break;
        }

        AudioChunk chunk;
        chunk.start_index = start_index;
        chunk.num_samples = end_index - start_index;
        chunk.data = &_sample_buffer[start_index];
        _chunks.push_back(chunk);

        start_index = end_index;
    }
}

uint32_t AudioInputModel::split_on_middle_silence(uint32_t start_index, uint32_t end_index) {
    auto mid_index = start_index + (end_index - start_index) / 2;

    auto count = (int)ceil((float)(end_index - mid_index) / _frame_length_samples);
    vector<bool> voices;

    // calculateVoiceActivityInChunks
#if USE_MODEL
    auto inputs = get_input_ptrs();
    memcpy(inputs[0].first, (char*)&_sample_buffer[mid_index], (count * _frame_length_samples) * sizeof(float));
    memcpy(inputs[1].first, &_energy_threshold, sizeof(float));

    invoke();

    auto outputs = get_output_ptrs();
#else
    auto frame_start = mid_index;
    uint32_t frame_end;
    voices.reserve(count);

    for (int idx = 0; idx < count; idx++) {
        frame_end = min(frame_start + _frame_length_samples, end_index);
        // calculate energy for the next 0.1 sec frame
        float energy = 0.0;
        for (int i = frame_start; i < frame_end; i++) {
            energy += (_sample_buffer[i] * _sample_buffer[i]);
        }
        auto avg_energy = sqrtf(energy / (frame_end - frame_start));
        voices.push_back((avg_energy > _energy_threshold));

        frame_start += _frame_length_samples;
    }
#endif

    // find the Longest Silence
    // all indices here mean voices index from above, not audio samples
    int longest_count = 0, longest_start = 0, longest_end = 0;
    int idx = 0, is_voice = 0;
#if USE_MODEL
    auto output_size = outputs[0].second / sizeof(float);
    auto output = reinterpret_cast<float*>(outputs[0].first);

    while (idx < output_size) {
        if (output[idx] > 0) {
#else
    while (idx < voices.size()) {
        if (voices[idx]) {
#endif
            idx++;
            is_voice++;
            continue;
        }
        // silence starts at idx
        auto endidx = idx;
#if USE_MODEL
        while (endidx < output_size && output[endidx] <= 0) {
#else
        while (endidx < voices.size() && !voices[endidx]) {
#endif
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
        return end_index;
    }

    auto silence_mid_idx = longest_start + (longest_end - longest_start) / 2;
    // voice activity index to audio sample index + mid_index
    return mid_index + silence_mid_idx * _frame_length_samples;
}