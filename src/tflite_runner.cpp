//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

#include <unistd.h>

#include <audio_input.hpp>
#include <post_proc.hpp>
#include <tflite_model.hpp>
#include "WhisperKit.h"

constexpr const char*  TFLITE_ROOT_PATH = "/sdcard/argmax/tflite";
constexpr const char*  DEFAULT_LIB_DIR = "/data/local/tmp/lib";
constexpr const char* DEFAULT_CACHE_DIR = "/data/local/tmp/cache";

#define TFLITE_INIT_CHECK(x)                              \
    if (!(x)) {                                           \
        LOGE("Error at %s:%d\n", __FUNCTION__, __LINE__); \
        exit(0);                                          \
    }

using namespace std;

unique_ptr<TFLiteModel> melspectro;
unique_ptr<TFLiteModel> encoder;
unique_ptr<TFLiteModel> decoder;
unique_ptr<AudioInputModel> audioinput;
unique_ptr<PostProcModel> postproc;

vector<pair<char*, int>> melspectro_inputs;
vector<pair<char*, int>> melspectro_outputs;
vector<pair<char*, int>> encoder_inputs;
vector<pair<char*, int>> encoder_outputs;
vector<pair<char*, int>> decoder_outputs;

vector<vector<int>> all_tokens;

chrono::time_point<chrono::high_resolution_clock> start_exec;

vector<string> read_input_list(const string inputListFile) {
    vector<string> inputFilePaths;
    string line;

    ifstream file(inputListFile);
    while (getline(file, line)) {
        LOGI("input file: %s\n", line.c_str());
        inputFilePaths.push_back(line);
    }

    return inputFilePaths;
}

bool status_returned_error(whisperkit_status_t status, const char* msg) {
    if (status != WHISPERKIT_STATUS_SUCCESS) {
        LOGE(msg);
        return true;
    }
    return false;
}

int tflite_init(vector<char*>& args) {
    if (args.size() < 2 || args[0] == nullptr || args[1] == nullptr) {
        return -1;
    }



    string audio_file = args[0];
    string tokenizer_json = string() + TFLITE_ROOT_PATH + "/inputs/converted_vocab.json";
    string audio_model = string() + TFLITE_ROOT_PATH + "/models/voice_activity_detection.tflite";
    string melspectro_model = string() + TFLITE_ROOT_PATH + "/models/melspectrogram.tflite";
    string encoder_model = string() + TFLITE_ROOT_PATH + "/models/encoder_" + args[1] + ".tflite";
    string decoder_model = string() + TFLITE_ROOT_PATH + "/models/decoder_" + args[1] + ".tflite";
    string postproc_model = string() + TFLITE_ROOT_PATH + "/models/postproc.tflite";

    melspectro = make_unique<TFLiteModel>("mel_spectrogram");
    encoder = make_unique<TFLiteModel>("whisper_encoder");
    decoder = make_unique<TFLiteModel>("whisper_decoder");

    postproc = make_unique<PostProcModel>(tokenizer_json);
    audioinput = make_unique<AudioInputModel>(audio_file);

    whisperkit_status_t status = WHISPERKIT_STATUS_SUCCESS;

    whisperkit_config_t* config;

    status = whisperkit_create_configuration(&config);

    if status_returned_error(status, "Failed to create configuration") {
        return -1;
    }

    status = whisperkit_configuration_set_audio_model(config, args[0]);
    if status_returned_error(status, "Failed to create configuration") {
        return -1;
    }
    config->set_audio_encoder(encoder_model.c_str());
    config->set_text_decoder(decoder_model.c_str());
    config->set_voice_activity_detector(audio_model.c_str());
    // TODO: allow setting single directory to indicate model paths (similar to repo arg in WhisperKit)

    // TODO: deprecate these methods on the config object.  We should
    config->set_tokenizer(tokenizer_json.c_str());
    config->set_audio_model(audio_model.c_str());
    config->set_melspectrogram_model(melspectro_model.c_str());

    config->set_postproc(postproc_model.c_str());

    string lib_dir(DEFAULT_LIB_DIR);
    if (args.size() > 2 && args[2] != nullptr) {
        lib_dir = string(args[2]);
    }
    string cache_dir(DEFAULT_CACHE_DIR);
    if (args.size() > 3 && args[3] != nullptr) {
        cache_dir = string(args[3]);
    }
    LOGI("lib dir: %s\ncache dir: %s\n", lib_dir.c_str(), cache_dir.c_str());

    TFLITE_INIT_CHECK(melspectro->initialize(melspectro_model, lib_dir, cache_dir, kHtpBackend));
    TFLITE_INIT_CHECK(encoder->initialize(encoder_model, lib_dir, cache_dir, kHtpBackend));
    TFLITE_INIT_CHECK(decoder->initialize(decoder_model, lib_dir, cache_dir, kHtpBackend));
    TFLITE_INIT_CHECK(audioinput->initialize(audio_model, lib_dir, cache_dir, kHtpBackend));
    TFLITE_INIT_CHECK(postproc->initialize(postproc_model, lib_dir, cache_dir, kUndefinedBackend));

    melspectro_inputs = melspectro->get_input_ptrs();
    melspectro_outputs = melspectro->get_output_ptrs();
    // outputs: melspectrogram
    assert(melspectro_outputs.size() == 1);

    encoder_inputs = encoder->get_input_ptrs();
    // retrieve encoder output tensor pointers
    encoder_outputs = encoder->get_output_ptrs();
    // outputs: k_cache, v_cache
    assert(encoder_outputs.size() == 2);

    decoder_outputs = decoder->get_output_ptrs();
    // outputs: logits, k_cache, v_cache
    assert(decoder_outputs.size() == 3);

    start_exec = chrono::high_resolution_clock::now();
    LOGI("tflite_init done..\n");
    return 0;
}

void tflite_close() {
    LOGI("decoded tokens: \n");

    for (auto& tokens : all_tokens) {
        for (auto& token : tokens) {
            cout << token << " ";
        }
        cout << endl;
    }
    cout << endl << endl;

    postproc->uninitialize();
    decoder->uninitialize();
    encoder->uninitialize();
    melspectro->uninitialize();
    audioinput->uninitialize();
    all_tokens.clear();
    LOGI("tflite_close done..\n");
}

unique_ptr<string> tflite_loop() {
    auto x = TOKEN_SOT;
    int index = 0;
    vector<int> tokens;
    auto out = make_unique<string>("");
    tokens.push_back(TOKEN_SOT);

    float timestamp = audioinput->get_next_chunk(melspectro_inputs[0].first);
    if (timestamp < 0) {
        return out;
    }

    // Perform melspectrogram inference
    melspectro->invoke(true);

    encoder->read_input_data(melspectro_outputs[0].first, 0);
    // Perform encoder inference
    encoder->invoke(true);

    // k_cache_cross
    decoder->read_input_data(encoder_outputs[0].first, 2);
    // v_cache_cross
    decoder->read_input_data(encoder_outputs[1].first, 3);
    // first k_cache_self is all zeros
    memset(decoder_outputs[1].first, 0, decoder_outputs[1].second);
    // first v_cache_self is all zeros
    memset(decoder_outputs[2].first, 0, decoder_outputs[2].second);

    for (; index < 224; index++) {
        // x
        decoder->read_input_data((char*)&x, 0);
        // index
        decoder->read_input_data((char*)&index, 1);
        // k_cache_self
        decoder->read_input_data(decoder_outputs[1].first, 4);
        // v_cache_self
        decoder->read_input_data(decoder_outputs[2].first, 5);

        decoder->invoke(true);

        x = postproc->process(index, reinterpret_cast<float*>(decoder_outputs[0].first),
                              (decoder_outputs[0].second / sizeof(float)), tokens, timestamp);

        tokens.push_back(x);
        if (x == TOKEN_EOT || x == -1) {
            break;
        }
    }
    all_tokens.push_back(tokens);
    out = postproc->get_sentence();

    return out;
}

unique_ptr<json> tflite_perfjson() {
    auto perfjson = make_unique<json>();
    (*perfjson)["audioinput"] = *audioinput->get_latency_json();
    (*perfjson)["melspectro"] = *melspectro->get_latency_json();
    (*perfjson)["encoder"] = *encoder->get_latency_json();
    (*perfjson)["decoder"] = *decoder->get_latency_json();
    (*perfjson)["postproc"] = *postproc->get_latency_json();

    auto end_exec = chrono::high_resolution_clock::now();
    float duration = chrono::duration_cast<std::chrono::microseconds>(end_exec - start_exec).count() / 1000.0;

    (*perfjson)["duration"] = ceil(duration * 100.0) / 100.0;
    return perfjson;
}

#ifndef ANDROID_JNI
int main(int argc, char* argv[]) {
    LOGI("Compiled " __DATE__ __TIME__
         "\n"
         "AXIE TFLite Runner\n"
         "Argmax, Inc.\n\n");
    if (argc < 3) {
        LOGE("Usage: %s <audio input> <tiny | base | small>\n", argv[0]);
        return -1;
    }

    vector<char*> args;
    for (int i = 0; i < argc; i++) {
        args.push_back(argv[i + 1]);
    }
    tflite_init(args);
    args.clear();

    while (true) {
        auto out = tflite_loop();
        if (out->empty()) {
            break;
        }
        LOGI("Text Out:\n%s\n\n", out->c_str());
    }

    auto perfjson = tflite_perfjson();
    LOGI(
        "Average model latency:\n"
        "  Audio Input:\t %.3f ms\n"
        "  Melspectro:\t %.3f ms\n"
        "  Encoder:\t %.3f ms\n"
        "  Decoder:\t %.3f ms\n"
        "  Postproc:\t %.3f ms\n"
        "=========================\n"
        "Total Duration:\t %.3f ms\n\n",
        (float)(*perfjson)["audioinput"]["avg"], (float)(*perfjson)["melspectro"]["avg"],
        (float)(*perfjson)["encoder"]["avg"], (float)(*perfjson)["decoder"]["avg"],
        (float)(*perfjson)["postproc"]["avg"], (float)(*perfjson)["duration"]);

    ofstream out_file;
    struct stat sb;

    if (stat(TFLITE_ROOT_PATH, &sb) != 0) {
        mkdir(TFLITE_ROOT_PATH, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    out_file.open(string() + TFLITE_ROOT_PATH + "/latencies.txt");
    out_file << perfjson->dump();
    out_file.close();

    tflite_close();

    return 0;
}
#endif