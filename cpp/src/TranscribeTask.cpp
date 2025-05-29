#include "TranscribeTask.hpp"

#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <array>
#include <ctime>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

// TODO : move these and all audio related code to a separate, non header file.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

#include "Models/TextDecoder.hpp"
#include "audio_input.hpp"
#include "backend_class.hpp"
#include "post_proc.hpp"
#include "tflite_msg.hpp"
// to be deleted
// JNI: set to app's cache dir
#if (JNI_BUILD)
#define TRANSCRIBE_TASK_TFLITE_ROOT_PATH "/sdcard/argmax/tflite"
#define TRANSCRIBE_TASK_DEFAULT_LIB_DIR "/data/local/tmp/lib"
#define TRANSCRIBE_TASK_DEFAULT_CACHE_DIR "/data/user/0/com.whispertflite/cache"
#elif (QNN_DELEGATE || GPU_DELEGATE)
// for Android QNN or GPU delegatea
#define TRANSCRIBE_TASK_TFLITE_ROOT_PATH "/sdcard/argmax/tflite"
#define TRANSCRIBE_TASK_DEFAULT_LIB_DIR "/data/local/tmp/lib"
#define TRANSCRIBE_TASK_DEFAULT_CACHE_DIR "/data/local/tmp/cache"
#else
#define TRANSCRIBE_TASK_TFLITE_ROOT_PATH "."
#define TRANSCRIBE_TASK_DEFAULT_LIB_DIR "./lib"
#define TRANSCRIBE_TASK_DEFAULT_CACHE_DIR "./cache"
#endif

#if defined(__ANDROID__)
#include <sys/system_properties.h>

std::string getProperty(const char* name) {
    char value[64] = {0};
    int len = __system_property_get(name, value);

    if (len > 0) {
        return std::move(std::string(value, len));
    }
    return {};
}
#endif

namespace WhisperKit::TranscribeTask {

// private class to encapsulate tflite related code
// so we can delete it imminently from whisperax_cli and whisperax.cpp
class Runtime {
   public:
    Runtime(const whisperkit_configuration_t& config);
    ~Runtime();

    void init();
    void close();
    int decoder_loop();
    void init_audio_input(int sample_rate = 16000, int num_channels = 1, int fmt = AV_SAMPLE_FMT_NONE);
    int append_audio_data(int size, char* pcm_buffer_left, char* pcm_buffer_right = nullptr);
    void conclude_transcription();

    void output_proc();
    bool check_qcom_soc();
    void audio_melspectro_proc();
    void encode_decode_postproc(float timestamp);
    void set_streaming_mode(bool streaming_mode);
    bool has_result_text();
    std::unique_ptr<std::string> get_result_text();
    void write_report(const char* audio_file, const std::string& transcription);

    std::unique_ptr<TFLiteMessenger> messenger;
    std::mutex gmutex;

   private:
    whisperkit_configuration_t config;
    std::string lib_dir;
    std::string cache_dir;
    std::string report_dir;
    float melspectro_timestamp;
    bool debug;
    bool is_qnn_backend;
    bool streaming_mode;

    std::unique_ptr<MODEL_SUPER_CLASS> melspectro;
    std::unique_ptr<MODEL_SUPER_CLASS> encoder;
    std::unique_ptr<TextDecoder> decoder;
    std::unique_ptr<AudioInputModel> audioinput;
    std::unique_ptr<PostProcModel> postproc;
    Tokenizer* tokenizer;

    std::vector<int> all_tokens;
    std::vector<std::string> all_msgs;
    std::vector<std::pair<char*, int>> melspectro_inputs;
    std::vector<std::pair<char*, int>> melspectro_outputs;
    std::vector<std::pair<char*, int>> encoder_inputs;
    std::vector<std::pair<char*, int>> encoder_outputs;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_exec;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_exec;
};

// copy pasted from audio_codec.hpp, which will be deleted
// Modifications: some forward declarations, and associated includes moved to TranscribeTask.cpp
using namespace std;

class AudioCodec {
   public:
    AudioCodec();
    ~AudioCodec() = default;

    bool open(const string filename, int verbose = 0);
    void close();

    AVFrame* get_frame() const { return _audio_frame; }
    int64_t get_duration_ms() const { return _duration; }
    bool is_running() const { return _is_running; }
    int get_datasize() const { return _frame_datasize; }
    int decode_pcm();
    bool is_streaming() { return _is_streaming; }

   private:
    AVIOContext* _io_context;
    unsigned char* _io_buffer;
    AVFormatContext* _format_context;
    AVCodecContext* _codec_context;
    AVCodec* _codec;
    AVFrame* _audio_frame;

    int _frame_datasize;
    string _codec_name;
    int64_t _duration;
    bool _is_running;
    bool _is_wav_input;
    bool _is_streaming;
};

}  // namespace WhisperKit::TranscribeTask

// for AudioCodec, now implemented inside WhisperKit::TranscribeTask

// TODO: Move AudioCodec related code to separate file, hide behind some interface
// related to input audio data (file or microphone source)
// For now, moved to here to remove from whisperax_cli and allow simpler runtime integration
// from [cli + library] -> [library and hidden behind some interface]
namespace WhisperKit::TranscribeTask {

#ifndef QCOM_SOC
#define QCOM_SOC "qcom"
#endif

#ifndef TFLITE_INIT_CHECK
#define TFLITE_INIT_CHECK(x)                              \
    if (!(x)) {                                           \
        LOGE("Error at %s:%d\n", __FUNCTION__, __LINE__); \
        exit(0);                                          \
    }
#endif

Runtime::Runtime(const whisperkit_configuration_t& config) { this->config = config; }

Runtime::~Runtime() { close(); }

void Runtime::set_streaming_mode(bool streaming_mode) { this->streaming_mode = streaming_mode; }

bool Runtime::check_qcom_soc() {
    vector<string> supported_socs{"SM8750", "SM8650", "SM8550", "SM8450", "SM8350"};
#if defined(__ANDROID__)
    auto soc = getProperty("ro.soc.model");
#else
    struct utsname info;
    std::string soc = "unknown";
    if (uname(&info) == 0) {
        soc = info.machine;
    }
#endif
    LOGI("SoC: \t%s", soc.c_str());
    if (find(begin(supported_socs), end(supported_socs), soc) != end(supported_socs)) {
        LOGI(" -> QNN HTP\n");
        return true;
    } else {  // all other SoCs
        LOGI(" -> TFLite GPU\n");
        return false;
    }
}

void Runtime::init() {
    lock_guard<mutex> lock(gmutex);

    // LOGI("tflite_init input: %s\n", config.get_model_path().c_str());

    int format = 0;

    is_qnn_backend = false;
#if (QNN_DELEGATE || GPU_DELEGATE)
    is_qnn_backend = check_qcom_soc();  // selecting runtime delegation for the model
#else
    LOGI("SoC: \tgeneric CPU (x86, arm64, etc) \n");
#endif

    std::string tokenizer_json = config.get_model_path() + "/tokenizer.json";
    std::string tokenizer_config_json = config.get_model_path() + "/config.json";
    std::string melspectro_model = config.get_model_path() + "/MelSpectrogram.tflite";
    std::string encoder_model = config.get_model_path() + "/AudioEncoder.tflite";
    std::string decoder_model = config.get_model_path() + "/TextDecoder.tflite";

    std::vector<std::string> required_files = {tokenizer_json, tokenizer_config_json, melspectro_model, encoder_model,
                                               decoder_model};
    for (const auto& file : required_files) {
        if (!std::filesystem::exists(file)) {
            LOGE("File does not exist: %s", file.c_str());
            std::stringstream ss;
            ss << file << " : required file not found";
            throw std::runtime_error(ss.str());
        }
    }

    melspectro = make_unique<MODEL_SUPER_CLASS>("mel_spectrogram");
    encoder = make_unique<MODEL_SUPER_CLASS>("whisper_encoder");

    decoder = TextDecoderFactory::CreateFromFile(decoder_model);

    // TODO move this to somewhere user accessible.
    tokenizer = tokenizer_init_from_file(tokenizer_json.c_str(), tokenizer_config_json.c_str());

    postproc = make_unique<PostProcModel>(tokenizer);

    lib_dir = std::string(TRANSCRIBE_TASK_DEFAULT_LIB_DIR);
    cache_dir = std::string(TRANSCRIBE_TASK_DEFAULT_CACHE_DIR);
    debug = config.get_verbose();
    if (!config.get_lib_dir().empty()) {
        lib_dir = config.get_lib_dir();
    }

    if (!config.get_cache_dir().empty()) {
        cache_dir = config.get_cache_dir();
    }

    if (!config.get_report_path().empty()) {
        report_dir = config.get_report_path();
    }

    TFLITE_INIT_CHECK(melspectro->initialize(melspectro_model, lib_dir, cache_dir, ComputeBackend::CPU, debug));
    TFLITE_INIT_CHECK(encoder->initialize(encoder_model, lib_dir, cache_dir, config.get_encoder_backend(), debug));
    TFLITE_INIT_CHECK(decoder->initialize(decoder_model, lib_dir, cache_dir, config.get_decoder_backend(), debug));
    TFLITE_INIT_CHECK(postproc->initialize(debug));

    melspectro_inputs = melspectro->get_input_ptrs();
    melspectro_outputs = melspectro->get_output_ptrs();
    // outputs: melspectrogram
    if (melspectro_outputs.size() != 1) throw std::invalid_argument("melspectro output tensor # has to be 1");

    encoder_inputs = encoder->get_input_ptrs();
    // retrieve encoder output tensor pointers
    encoder_outputs = encoder->get_output_ptrs();
    // outputs: k_cache, v_cache
    if (encoder_outputs.size() != 2) throw std::invalid_argument("audio encoder output tensor # has to be 2");

    all_tokens.clear();
    all_tokens.reserve(1 << 18);  // max 256K tokens
    all_msgs.clear();
    all_msgs.reserve(1 << 14);  // max 4096 sentences

    messenger = std::make_unique<TFLiteMessenger>();
    messenger->_running = true;
}

void Runtime::init_audio_input(int sample_rate, int num_channels, int fmt) {
    audioinput = make_unique<AudioInputModel>(sample_rate, num_channels, fmt);

    TFLITE_INIT_CHECK(audioinput->initialize(debug));

    start_exec = chrono::high_resolution_clock::now();
}

void Runtime::conclude_transcription() {
    lock_guard<mutex> lock(gmutex);

    messenger->_running = false;

    end_exec = chrono::high_resolution_clock::now();
}

bool Runtime::has_result_text() { return !all_msgs.empty(); }

std::unique_ptr<std::string> Runtime::get_result_text() {
    auto output = make_unique<std::string>();
    if (all_msgs.empty()) {
        return output;
    }

    for (std::vector<std::string>::const_iterator iter = all_msgs.begin(); iter != all_msgs.end(); ++iter) {
        *output += (*iter + '\n');
    }

    all_msgs.clear();
    return output;
}

void Runtime::close() {
    tokenizer_free(tokenizer);
    tokenizer = nullptr;
    postproc->uninitialize();
    decoder->uninitialize();
    encoder->uninitialize();
    melspectro->uninitialize();
    audioinput->uninitialize();
}

int Runtime::decoder_loop() {
    lock_guard<mutex> lock(gmutex);

    while (true) {
        audio_melspectro_proc();
        if (melspectro_timestamp < 0) {
            return -1;
        }

        encode_decode_postproc(melspectro_timestamp);
    }

    return 0;
}

void Runtime::audio_melspectro_proc() {
    melspectro_timestamp = audioinput->get_next_chunk(melspectro_inputs[0].first);

    if (melspectro_timestamp < 0) {
        return;
    }

    encoder->get_mutex()->lock();
    melspectro->invoke(true);
    encoder->get_mutex()->unlock();
}

void Runtime::encode_decode_postproc(float timestamp) {
    auto x = tokenizer->specialTokens.startOfTranscriptToken;
    int index = 0;
    vector<int> tokens;
    tokens.push_back(tokenizer->specialTokens.startOfTranscriptToken);

    encoder->get_mutex()->lock();
    encoder->read_input_data(melspectro_outputs[0].first, 0);
    encoder->get_mutex()->unlock();
    encoder->invoke(true);

    auto k_cache_cross = encoder->get_output_with_name("k_cache_cross");
    if (k_cache_cross.first == nullptr) {
        k_cache_cross = encoder->get_output_with_name("k_cache");
    }
    auto v_cache_cross = encoder->get_output_with_name("v_cache_cross");
    if (v_cache_cross.first == nullptr) {
        v_cache_cross = encoder->get_output_with_name("v_cache");
    }

    if (k_cache_cross.first == nullptr || v_cache_cross.first == nullptr) {
        LOGE("Failed to get k_cache_cross or v_cache_cross");
        return;
    }

    decoder->bind_input_tensor(k_cache_cross.first, "k_cache_cross");
    decoder->bind_input_tensor(v_cache_cross.first, "v_cache_cross");

    decoder->initialize_kv_cache();

    constexpr const int MAX_DECODING_STEPS = 224;
    for (; index < MAX_DECODING_STEPS; index++) {
        decoder->bind_input_tensor((char*)&x, "x");
        decoder->bind_input_tensor((char*)&index, "index");
        decoder->update_kv_cache();

        decoder->invoke(true);

        const auto& logits_tensor = decoder->get_logits_tensor();
        const auto& logits = reinterpret_cast<float*>(logits_tensor.first);
        const auto& logits_size = logits_tensor.second / sizeof(float);

        x = postproc->process(index, logits, logits_size, tokens, timestamp);

        tokens.push_back(x);
        all_tokens.push_back(x);
        if (x == tokenizer->specialTokens.endOfTranscriptToken || x == -1) {
            postproc->decode_segment(tokens);
            break;
        }
    }

    messenger->_msg = postproc->get_sentence();
    messenger->_timestamp = timestamp;
    messenger->_cond_var.notify_all();
    all_msgs.push_back(messenger->get_message());
}

int Runtime::append_audio_data(int size, char* pcm_buffer0, char* pcm_buffer1) {
    if (!pcm_buffer0 || size <= 0) {
        return -1;
    }
    audioinput->fill_pcmdata(size, pcm_buffer0, pcm_buffer1);
    return audioinput->get_curr_buf_time();
}

void Runtime::write_report(const char* audio_file, const std::string& transcription) {
    if (report_dir.empty()) return;

    auto testjson = make_unique<json>();
    auto testinfo = json();
    auto staticattr = json();
    auto latstats = json();
    auto measure = json();
    auto timings = json();

    testinfo["model"] = config.get_model_path();
#if defined(__ANDROID__)
    testinfo["device"] = getProperty("ro.product.brand") + " " + getProperty("ro.soc.model");
#else
    struct utsname info;
    if (uname(&info) == 0) {
        testinfo["device"] = std::string("Architecture: ") + info.machine;
    } else {
        testinfo["device"] = "Unknown architecture";
    }
#endif

    time_t now;
    char buf[32];
    time(&now);
    strftime(buf, sizeof buf, "%FT%TZ", gmtime(&now));
    testinfo["date"] = buf;

    float duration = chrono::duration_cast<std::chrono::microseconds>(end_exec - start_exec).count() / 1000.0;

    // Extract filename from path using string operations
    std::string filename = audio_file;
    size_t pos = filename.find_last_of("/\\");
    std::string basename = (pos != std::string::npos) ? filename.substr(pos + 1) : filename;
    testinfo["audioFile"] = basename;
    // testinfo["prediction"] = all_tokens;  -> for numeric tokens output
    testinfo["prediction"] = transcription;

    timings["inputAudioSeconds"] = audioinput->get_total_input_time();
    timings["totalEncodingRuns"] = encoder->get_inference_num();
    // TODO: get the right number once temp fallback is implemented
    timings["totalDecodingFallbacks"] = 0;
    timings["totalDecodingLoops"] = decoder->get_inference_num();
    timings["fullPipeline"] = duration;
    testinfo["timings"] = timings;

#if defined(__ANDROID__)
    staticattr["os"] = "Android " + getProperty("ro.build.version.release");
#else
    if (uname(&info) == 0) {
        staticattr["os"] = info.release;
    } else {
        staticattr["os"] = "Unknown OS";
    }
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

    ofstream out_file;
    struct stat sb;

    if (stat(report_dir.c_str(), &sb) != 0) {
        mkdir(report_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    out_file.open(report_dir + "/output.json");
    out_file << testjson->dump() << endl;
    out_file.close();
}

constexpr const uint64_t INPUT_BUFFER_SIZE = (8 << 20);
constexpr const uint64_t STREAM_READ_SIZE = (512 << 10);  // has to be larger than 128KB

static int cbDecodeInterrupt(void* ctx) {
    // return whether to stop the input stream or not
    AudioCodec* codec = (AudioCodec*)ctx;
    if (codec->is_running())
        return 0;
    else
        return 1;
}

//=========== AudioCodec =================
AudioCodec::AudioCodec() {
    _format_context = nullptr;
    _io_context = nullptr;
    _io_buffer = nullptr;
    _codec_context = nullptr;
    _codec = nullptr;
    _audio_frame = nullptr;
    _is_wav_input = false;
    _frame_datasize = 0;
    _is_streaming = false;
}

bool AudioCodec::open(string filename, int verbose) {
    int ret;
    AVCodecParameters* codec_par = nullptr;
    AVDictionary** opts = nullptr;
    AVDictionary* format_opts = nullptr;

    if (!verbose) {
        av_log_set_level(AV_LOG_ERROR);
    }

    // allocating Format I/O context
    _format_context = avformat_alloc_context();
    _audio_frame = av_frame_alloc();

    if (!_format_context || !_audio_frame) {
        LOGE("alloc format or audio frame context failed\n");
        return false;
    }

    av_dict_set(&format_opts, "scan_all_pmts", "1", AV_DICT_DONT_OVERWRITE);

    _format_context->interrupt_callback.callback = cbDecodeInterrupt;
    _format_context->interrupt_callback.opaque = this;
    _format_context->max_analyze_duration = 1024000;
    _is_running = true;

    AVInputFormat* input_format = nullptr;

    if (strstr(filename.c_str(), "http://") || strstr(filename.c_str(), "tcp://")) {
        av_dict_set(&format_opts, "listen", "0", 0);
        av_dict_set(&format_opts, "timeout", "20000000", 0);
        _is_streaming = true;
    }
    ret = avformat_open_input(&_format_context, filename.c_str(), input_format, &format_opts);
    if (ret < 0) {
        LOGE("avformat_open_input Error: %s\n", *av_err2string(ret));
        return false;
    }

    opts = (AVDictionary**)av_calloc(_format_context->nb_streams, sizeof(*opts));
    avformat_find_stream_info(_format_context, opts);
    _duration = (_format_context->duration) / 1000;

    for (unsigned int i = 0; i < _format_context->nb_streams; i++) {
        codec_par = _format_context->streams[i]->codecpar;
        if (codec_par->codec_type == AVMEDIA_TYPE_AUDIO) break;
    }
    if (codec_par == nullptr) throw std::invalid_argument("codec_par is a null ptr..");

    _audio_frame->sample_rate = codec_par->sample_rate;
    _audio_frame->ch_layout = codec_par->ch_layout;
    if (codec_par->format == AV_SAMPLE_FMT_NONE)
        _audio_frame->format = AV_SAMPLE_FMT_S16;
    else
        _audio_frame->format = codec_par->format;

    if (filename.find(".wav") != string::npos || filename.find(".wave") != string::npos) {
        _is_wav_input = true;
    } else {
        _codec = (AVCodec*)avcodec_find_decoder(codec_par->codec_id);
        if (!_codec) {
            LOGE("avcodec_find_decoder failed\n");
            return false;
        }
        _codec_name = string(avcodec_get_name(codec_par->codec_id));

        // LOGI("Audio Codec: %s\n", _codec_name.c_str());
        if (_codec_context != nullptr) {
            avcodec_free_context(&_codec_context);
        }
        _codec_context = avcodec_alloc_context3(_codec);

        avcodec_parameters_to_context(_codec_context, codec_par);

        if (avcodec_open2(_codec_context, _codec, opts) < 0) {
            LOGE("Could not open audio codec..\n");
            return false;
        }
    }

    if (verbose > 0) {
        av_dump_format(_format_context, 0, nullptr, false);
    }
    av_free(opts);

    return true;
}

void AudioCodec::close() {
    if (_codec_context) {
        av_free(_codec_context);
        _codec_context = nullptr;
    }
    if (_audio_frame) {
        av_frame_unref(_audio_frame);
        av_frame_free(&_audio_frame);
        _audio_frame = nullptr;
    }
    if (_io_context) {
        avio_flush(_io_context);
        av_freep(&_io_context->buffer);  // note that it is referencing m_pIOBuffer
        av_freep(&_io_context);
        _io_buffer = nullptr;
    }
    if (_format_context) {
        avformat_close_input(&_format_context);
        avformat_free_context(_format_context);
        _format_context = nullptr;
    }
    _is_running = false;
}

int AudioCodec::decode_pcm() {
    int retry = 0;
    AVPacket packet;
    int ret = av_read_frame(_format_context, &packet);
    if (ret < 0) {
        return ret;
    }

    if (_is_wav_input && packet.size > 0) {
        _audio_frame->data[0] = packet.data;
        _audio_frame->nb_samples = packet.size / av_get_bytes_per_sample((AVSampleFormat)_audio_frame->format);
        _frame_datasize = packet.size;
        return 0;
    }

    if (packet.size > 0) {
        ret = avcodec_send_packet(_codec_context, &packet);
        if (ret < 0) {
            LOGE("Error sending a packet: %s\n", *av_err2string(ret));
            return ret;
        }
    }
    av_frame_unref(_audio_frame);
    ret = avcodec_receive_frame(_codec_context, _audio_frame);
    if (ret == AVERROR(EAGAIN)) {
        return ret;
    } else if (ret == AVERROR_EOF) {
        return ret;
    } else if (ret < 0) {
        LOGE("Error during decoding: %s\n", *av_err2string(ret));
        return ret;
    }

    _frame_datasize = _audio_frame->nb_samples * av_get_bytes_per_sample((AVSampleFormat)_audio_frame->format);

    return 0;
}

#ifdef QCOM_SOC
#undef QCOM_SOC
#endif

}  // namespace WhisperKit::TranscribeTask

using namespace WhisperKit::TranscribeTask;

TranscribeTask::TranscribeTask(const whisperkit_configuration_t& config) {
    this->config = config;

    audio_codec = make_unique<AudioCodec>();
    argsjson = make_unique<nlohmann::json>();

    runtime = std::make_unique<Runtime>(config);
    runtime->init();
}

TranscribeTask::~TranscribeTask() { runtime->close(); }

void TranscribeTask::textOutputProc() {
    text_out_thread = make_unique<thread>([this]() {
        unique_lock<mutex> ulock(runtime->messenger->_mutex);

        runtime->messenger->_cond_var.wait(ulock, [this] { return !runtime->messenger->_running; });
    });
}

void TranscribeTask::transcribe(const char* audio_file, whisperkit_transcription_result_t* transcription_result) {
    _transcription = transcription_result;
    if (!audio_codec->open(audio_file, config.get_verbose())) {
        LOGE("Error opening audio file: %s\n", audio_file);
        throw std::runtime_error("Error opening audio file");
    }
    runtime->set_streaming_mode(false);

    auto audio_frame = audio_codec->get_frame();
    if (audio_frame == nullptr) {
        throw std::invalid_argument("audio frame is null");
        return;
    }

    runtime->init_audio_input(audio_frame->sample_rate, audio_frame->ch_layout.nb_channels, audio_frame->format);

    int pcm_secs = 0, ret = 0;
    int segment_length = audio_codec->is_streaming() ? 15 : 30;
    bool transcribed = false;

    while (ret != AVERROR_EOF) {
        ret = audio_codec->decode_pcm();
        if (ret < 0 || audio_codec->get_datasize() == 0) {
            usleep(100000);  // 100ms
            continue;
        }

        transcribed = appendAudio(audio_codec->get_datasize(), (char*)audio_codec->get_frame()->data[0],
                                  (char*)audio_codec->get_frame()->data[1]);
    }
    closeStreaming();
    LOGI("Transcription #%d (final): %s\n", chunk_idx++, _transcription->get_chunk_transcription().c_str());

    runtime->write_report(audio_file, _transcription->get_transcription());
}

void TranscribeTask::initStreaming(whisperkit_transcription_result_t* transcription_result, int sample_rate,
                                   int num_channels) {
    _transcription = transcription_result;
    chunk_idx = 0;

    runtime->init_audio_input(sample_rate, num_channels);
    runtime->set_streaming_mode(true);
}

bool TranscribeTask::appendAudio(int size, char* buffer0, char* buffer1) {
    if (size <= 0 || buffer0 == nullptr) return false;

    const int segment_length = 30;  // one chunk of audio length

    int pcm_secs = runtime->append_audio_data(size, buffer0, buffer1);
    if (pcm_secs < segment_length) {
        return false;
    }

    runtime->decoder_loop();

    auto result_text = runtime->get_result_text();
    if (_transcription != nullptr && !result_text->empty()) {
        _transcription->set_transcription(*result_text);
    }

    LOGI("Transcription #%d (ongoing): %s\n", chunk_idx++, _transcription->get_chunk_transcription().c_str());
    return true;
}

void TranscribeTask::closeStreaming() {
    runtime->decoder_loop();
    runtime->conclude_transcription();

    auto result_text = runtime->get_result_text();
    if (_transcription != nullptr) {
        _transcription->set_transcription(*result_text);
    }
}
