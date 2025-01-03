//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#include <ctime>
#include <post_proc.hpp>
#include <audio_input.hpp>
#include <whisperax.hpp>

#define QCOM_SOC    "qcom"
#define TFLITE_INIT_CHECK(x)                          \
    if (!(x)) {                               \
        LOGE("Error at %s:%d\n", __FUNCTION__, __LINE__); \
        exit(0);                                         \
    }

using namespace std;

mutex gmutex;

unique_ptr<thread> encode_decode_thread;
shared_ptr<TFLiteMessenger> messenger;

unique_ptr<MODEL_SUPER_CLASS> melspectro;
unique_ptr<MODEL_SUPER_CLASS> encoder;
unique_ptr<MODEL_SUPER_CLASS> decoder;
unique_ptr<AudioInputModel> audioinput;
unique_ptr<PostProcModel> postproc;

vector<int> all_tokens;
vector<pair<char*, int>> melspectro_inputs;
vector<pair<char*, int>> melspectro_outputs;
vector<pair<char*, int>> encoder_inputs;
vector<pair<char*, int>> encoder_outputs;
vector<pair<char*, int>> decoder_outputs;

chrono::time_point<chrono::high_resolution_clock> start_exec;


unique_ptr<string> cmdexec(const char* cmd) {
    array<char, 128> buffer;
    auto result = make_unique<string>();
    unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        (*result) += buffer.data();
    }
    while (result->find('\n') != string::npos){
        auto pos = result->find('\n');
        result->erase(result->begin() + pos, result->begin()+pos+1);
    }
    
    return result;
}

bool check_qcom_soc(){
    vector<string> supported_socs{
        "SM8650", "SM8550", "SM8450","SM8350"
    };

    auto soc = *cmdexec("getprop ro.soc.model");
    LOGI("SoC: \t%s", soc.c_str());
    if (find(begin(supported_socs), end(supported_socs), soc) 
        != end(supported_socs)){
        LOGI(" -> QNN HTP\n");
        return true; 
    } else { // all other SoCs
        LOGI(" -> TFLite GPU\n");
        return false; 
    }
}

int tflite_init(string argstr){
    if (argstr.size() < 8) {
        return -1;
    }

    lock_guard<mutex> lock(gmutex);

    LOGI("tflite_init input: %s\n", argstr.c_str());
    auto args = json::parse(argstr);

    string model_size = args["size"];
    assert(args.contains("freq"));
    assert(args.contains("ch"));

    int format = 0;
    if (args.contains("fmt"))
    {
        format = args["fmt"];
    }
    string root_path = TFLITE_ROOT_PATH;
    if (args.contains("root_path"))
    {
        root_path = args["root_path"];
    }

    int backend = kUndefinedBackend;
#if (defined(QNN_DELEGATE) || defined(GPU_DELEGATE)) 
    if (check_qcom_soc()) // selecting runtime delegation for the model
        backend = kHtpBackend; 
#else
    LOGI("SoC: \tgeneric CPU (x86, arm64, etc) \n");
#endif

    audioinput = make_unique<AudioInputModel>(
        (int)args["freq"], (int)args["ch"], format
    );

    std::string tokenizer_json = root_path +
        "/openai_whisper-" + model_size + "/converted_vocab.json";
    std::string audio_model = root_path +
        "/openai_whisper-" + model_size + "/voice_activity_detection.tflite";
    std::string melspectro_model = root_path +
        "/openai_whisper-" + model_size + "/MelSpectrogram.tflite";
    std::string encoder_model = root_path +
        "/openai_whisper-" + model_size + "/AudioEncoder.tflite";
    std::string decoder_model = root_path +
        "/openai_whisper-" + model_size + "/TextDecoder.tflite";
    std::string postproc_model = root_path +
        "/openai_whisper-" + model_size + "/postproc.tflite";

    melspectro = make_unique<MODEL_SUPER_CLASS>("mel_spectrogram");
    encoder = make_unique<MODEL_SUPER_CLASS>("whisper_encoder");
    decoder = make_unique<MODEL_SUPER_CLASS>("whisper_decoder");
    postproc = make_unique<PostProcModel>(tokenizer_json);

    string lib_dir(DEFAULT_LIB_DIR); 
    if (args.contains("lib")) {
        lib_dir = args["lib"];
    }
    string cache_dir(DEFAULT_CACHE_DIR); 
    if (args.contains("cache")) {
        cache_dir = args["cache"];
    }
    bool debug = false; 
    if (args.contains("debug")) {
        debug = args["debug"];
    }

    LOGI("root dir:\t%s\nlib dir:\t%s\ncache dir:\t%s\n", 
        root_path.c_str(), lib_dir.c_str(), cache_dir.c_str()
    );

    TFLITE_INIT_CHECK(melspectro->initialize(
        melspectro_model, lib_dir, cache_dir, backend, debug
    ));
    TFLITE_INIT_CHECK(encoder->initialize(
        encoder_model, lib_dir, cache_dir, backend, debug
    ));
    TFLITE_INIT_CHECK(decoder->initialize(
        decoder_model, lib_dir, cache_dir, backend, debug
    ));
    TFLITE_INIT_CHECK(audioinput->initialize(
        audio_model, lib_dir, cache_dir, backend, debug
    ));
    TFLITE_INIT_CHECK(postproc->initialize(
        postproc_model, lib_dir, cache_dir, kUndefinedBackend, debug
    ));

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

    all_tokens.clear();
    all_tokens.reserve(1 << 18); // max 256K tokens

    start_exec = chrono::high_resolution_clock::now();

    messenger = make_shared<TFLiteMessenger>();
    messenger->_running = true;
    LOGI("tflite_init done..\n");

    return 0;
}

int tflite_close(){
    lock_guard<mutex> lock(gmutex);

    messenger->_running = false;
    if (encode_decode_thread.get() != nullptr && 
        encode_decode_thread->joinable()) {
        encode_decode_thread->join();
    }
    
    postproc->uninitialize();
    decoder->uninitialize();
    encoder->uninitialize();
    melspectro->uninitialize();
    audioinput->uninitialize();
    LOGI("tflite_close done..\n");
    return 0;
}

int tflite_write_data(char* pcm_buffer, int size){
    audioinput->fill_pcmdata(size, pcm_buffer);

    return audioinput->get_curr_buf_time();
}

void encode_decode_postproc(float timestamp)
{
    auto x = TOKEN_SOT; 
    int index = 0;
    vector<int> tokens;
    tokens.push_back(TOKEN_SOT);

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

    for(; index < 224; index++){
        // x
        decoder->read_input_data((char*)&x, 0);
        // index
        decoder->read_input_data((char*)&index, 1);
        // k_cache_self
        decoder->read_input_data(decoder_outputs[1].first, 4);
        // v_cache_self
        decoder->read_input_data(decoder_outputs[2].first, 5);

        decoder->invoke(true);

        x = postproc->process(
            index, 
            reinterpret_cast<float*>(decoder_outputs[0].first), 
            (decoder_outputs[0].second / sizeof(float)),
            tokens, 
            timestamp
        );

        tokens.push_back(x);
        all_tokens.push_back(x);
        if( x == TOKEN_EOT || x == -1){
            break;
        }
    }
    messenger->_msg = postproc->get_sentence();
    messenger->_timestamp = timestamp;
    messenger->_cond_var.notify_all();
}

int tflite_loop()
{
    lock_guard<mutex> lock(gmutex);

    // parallel execution of 2 threads for a pipeline of
    // 1) audio input & melspectrogram,
    // 2) encoder & decoder & postproc
    float timestamp = audioinput->get_next_chunk(
        melspectro_inputs[0].first
    );
    if (timestamp < 0) {
        return -1; 
    }
    encoder->get_mutex()->lock();
    melspectro->invoke(true);
    encoder->get_mutex()->unlock();

    if (encode_decode_thread.get() != nullptr && 
        encode_decode_thread->joinable()) {
        encode_decode_thread->join();
    }

    encode_decode_thread = make_unique<thread>(encode_decode_postproc, timestamp);
    return 0;
}

unique_ptr<json> tflite_perfjson(){
    auto perfjson = make_unique<json>(); 
    (*perfjson)["audioinput"] =  *audioinput->get_latency_json();
    (*perfjson)["melspectro"] = *melspectro->get_latency_json();
    (*perfjson)["encoder"] = *encoder->get_latency_json();
    (*perfjson)["decoder"] = *decoder->get_latency_json();
    (*perfjson)["postproc"] = *postproc->get_latency_json();

    auto end_exec = chrono::high_resolution_clock::now();
    float duration =
        chrono::duration_cast<std::chrono::milliseconds>(
            end_exec - start_exec).count();

    (*perfjson)["duration"] = ceil(duration * 100.0) / 100.0;
    return perfjson;
}

unique_ptr<json> get_test_json(
    const char* audiofile, 
    const char* model_size, 
    float duration
) {
    auto testjson = make_unique<json>();
    auto testinfo = json(); 
    auto staticattr = json();
    auto latstats = json();
    auto measure = json();
    auto timings = json();

    testinfo["model"] = string("openai_whisper-") + model_size;
#if (defined(QNN_DELEGATE) || defined(GPU_DELEGATE)) 
    testinfo["device"] = *cmdexec("getprop ro.product.brand") + " " 
                        + *cmdexec("getprop ro.soc.model");
#else
    testinfo["device"] = *cmdexec("lscpu | grep Architecture");
#endif
    time_t now;
    char buf[32];
    time(&now);
    strftime(buf, sizeof buf, "%FT%TZ", gmtime(&now));
    testinfo["date"] = buf;

    filesystem::path audio_path(audiofile); 
    testinfo["audioFile"] = audio_path.filename().string();
    testinfo["prediction"] = all_tokens; 
    testinfo["timeElapsedInSeconds"] = duration;

    timings["inputAudioSeconds"] = audioinput->get_total_input_time();
    timings["totalEncodingRuns"] = encoder->get_inference_num();
    // TODO: get the right number once temp fallback is implemented
    timings["totalDecodingFallbacks"] = 0;
    timings["totalDecodingLoops"] = decoder->get_inference_num();
    timings["fullPipeline"] = (audioinput->get_latency_sum()
                                + melspectro->get_latency_sum()
                                + encoder->get_latency_sum()
                                + decoder->get_latency_sum()
                                + postproc->get_latency_sum());
    testinfo["timings"] = timings;

#if (defined(QNN_DELEGATE) || defined(GPU_DELEGATE)) 
    staticattr["os"] = "Android " + *cmdexec("getprop ro.build.version.release");
#else
    staticattr["os"] = *cmdexec("uname -r");
#endif

    measure["cumulativeTokens"] = all_tokens.size();
    measure["numberOfMeasurements"] = all_tokens.size();
    measure["timeElapsed"] = decoder->get_latency_sum();
    
    latstats["measurements"] = measure; 
    latstats["totalNumberOfMeasurements"] = all_tokens.size();
    latstats["units"] = "Tokens/Sec";

    (*testjson)["latencyStats"] = latstats;
    (*testjson)["testInfo"] = testinfo;
    (*testjson)["staticAttributes"] = staticattr;

    return testjson;
}
