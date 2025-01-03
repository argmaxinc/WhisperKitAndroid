#include "TranscribeTask.h"

#include "whisperax.hpp"
#include <memory>

// TODO : move these and all audio related code to a separate, non header file.
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/time.h>
#include <libavutil/file.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>

#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

#include "tflite_msg.hpp"
#include "backend_class.hpp"
#include "audio_input.hpp"
#include "post_proc.hpp"

namespace WhisperKit::TranscribeTask {


// private class to encapsulate tflite related code
// so we can delete it imminently from whisperax_cli and whisperax.cpp
class Runtime {
    public:
        Runtime(const whisperkit_configuration_t& config);
        ~Runtime();

        void tflite_init_priv();
        void tflite_close_priv();
        void tflite_loop_priv();
        void tflite_perfjson_priv();
        void text_output_proc_priv();
        void encode_decode_postproc_priv(float timestamp);
        int tflite_write_data_priv(int size, char* pcm_buffer);
        void tflite_init_audioinput(const AudioCodec* audio_codec, const char* audio_file);
        std::shared_ptr<TFLiteMessenger> messenger;
        std::mutex gmutex;

    private:
        whisperkit_configuration_t config;
        std::unique_ptr<std::thread> encode_decode_thread;
        std::string lib_dir;
        std::string cache_dir;
        bool debug;
        int backend;
        std::unique_ptr<MODEL_SUPER_CLASS> melspectro;
        std::unique_ptr<MODEL_SUPER_CLASS> encoder;
        std::unique_ptr<MODEL_SUPER_CLASS> decoder;
        std::unique_ptr<AudioInputModel> audioinput;
        std::unique_ptr<PostProcModel> postproc;

        std::vector<int> all_tokens;
        std::vector<std::pair<char*, int>> melspectro_inputs;
        std::vector<std::pair<char*, int>> melspectro_outputs;
        std::vector<std::pair<char*, int>> encoder_inputs;
        std::vector<std::pair<char*, int>> encoder_outputs;
        std::vector<std::pair<char*, int>> decoder_outputs;

        std::chrono::time_point<std::chrono::high_resolution_clock> start_exec;
};

// copy pasted from audio_codec.hpp, which will be deleted
// Modifications: some forward declarations, and associated includes moved to TranscribeTask.cpp
using namespace std;

class AudioCodec
{
public:
    AudioCodec();
    ~AudioCodec();

    bool    open(string filename);
    void    close();

    int             get_samplerate() const { return _freq; }
    int             get_channel() const { return _channel; }
    int             get_format() const;
    AVFrame*        get_frame() const { return _dst_frame; }
    int64_t         get_duration_ms() const { return _duration; }
    bool            is_running() const { return _is_running; }
    int             get_datasize() const { return _single_ch_datasize; }
    int             decode_pcm();

private:
    AVIOContext*    _io_context;
    unsigned char*  _io_buffer;
    AVFormatContext *_format_context;
    AVCodecContext  *_codec_context;
    AVCodec         *_codec;
    AVFrame         *_src_frame;
    AVFrame         *_dst_frame;
    SwrContext      *_swr;

    int             _freq;
    int             _single_ch_datasize;
    int             _channel;
    AVSampleFormat  _sample_format;
    string          _codec_name;
    int64_t         _duration;
    bool            _is_running;
    bool            _is_wav_input; 
};

}

// for AudioCodec, now implemented inside WhisperKit::TranscribeTask

// TODO: Move AudioCodec related code to separate file, hide behind some interface 
// related to input audio data (file or microphone source)
// For now, moved to here to remove from whisperax_cli and allow simpler runtime integration 
// from [cli + library] -> [library and hidden behind some interface]
namespace WhisperKit::TranscribeTask {


#ifndef QCOM_SOC
#define QCOM_SOC    "qcom"
#endif

#ifndef TFLITE_INIT_CHECK
#define TFLITE_INIT_CHECK(x)                          \
    if (!(x)) {                               \
        LOGE("Error at %s:%d\n", __FUNCTION__, __LINE__); \
        exit(0);                                         \
    }
#endif



Runtime::Runtime(const whisperkit_configuration_t& config) {
    this->config = config;
}

Runtime::~Runtime() {
    tflite_close_priv();
}

void Runtime::text_output_proc_priv() {

}

void Runtime::tflite_init_priv() {

    lock_guard<mutex> lock(gmutex);

    LOGI("tflite_init input: %s\n", config.get_model_path().c_str());

    int format = 0;

    backend = kUndefinedBackend;
#if (defined(QNN_DELEGATE) || defined(GPU_DELEGATE)) 
    if (check_qcom_soc()) // selecting runtime delegation for the model
        backend = kHtpBackend; 
#else
    LOGI("SoC: \tgeneric CPU (x86, arm64, etc) \n");
#endif

    // TODO: this should be using std::filesystem..
    std::string tokenizer_json = config.get_model_path() + "/converted_vocab.json";
    std::string melspectro_model =  config.get_model_path() +  "/MelSpectrogram.tflite";
    std::string encoder_model =  config.get_model_path() + "/AudioEncoder.tflite";
    std::string decoder_model =  config.get_model_path() +  "/TextDecoder.tflite";
    std::string postproc_model =  config.get_model_path() +  "/postproc.tflite";

    melspectro = make_unique<MODEL_SUPER_CLASS>("mel_spectrogram");
    encoder = make_unique<MODEL_SUPER_CLASS>("whisper_encoder");
    decoder = make_unique<MODEL_SUPER_CLASS>("whisper_decoder");
    postproc = make_unique<PostProcModel>(tokenizer_json);

    lib_dir = std::string(DEFAULT_LIB_DIR); 
    cache_dir = std::string(DEFAULT_CACHE_DIR); 
    debug = config.get_verbose();
    if (!config.get_lib_dir().empty()) {
        lib_dir = config.get_lib_dir();
    }

    if (!config.get_cache_dir().empty()) {
        cache_dir = config.get_cache_dir();
    }

    LOGI("root dir:\t%s\nlib dir:\t%s\ncache dir:\t%s\n", 
        config.get_model_path().c_str(), lib_dir.c_str(), cache_dir.c_str()
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
}

void Runtime::tflite_init_audioinput(const AudioCodec* audio_codec, const char* audio_file) {

    const int freq = audio_codec->get_samplerate();
    const int channels = audio_codec->get_channel(); 
    const int fmt = audio_codec->get_format(); 
    audioinput = make_unique<AudioInputModel>(freq, channels, fmt);

    std::string audio_model =  config.get_model_path() +  "/voice_activity_detection.tflite";

    TFLITE_INIT_CHECK(audioinput->initialize(
        audio_model, lib_dir, cache_dir, backend, debug
    ));

}

void Runtime::tflite_close_priv() {
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
}
void Runtime::tflite_loop_priv() {
    lock_guard<mutex> lock(gmutex);

    // parallel execution of 2 threads for a pipeline of
    // 1) audio input & melspectrogram,
    // 2) encoder & decoder & postproc
    float timestamp = audioinput->get_next_chunk(
        melspectro_inputs[0].first
    );
    if (timestamp < 0) {
        throw std::runtime_error("Error getting next chunk");
    }
    encoder->get_mutex()->lock();
    melspectro->invoke(true);
    encoder->get_mutex()->unlock();

    if (encode_decode_thread.get() != nullptr && 
        encode_decode_thread->joinable()) {
        encode_decode_thread->join();
    }


    encode_decode_thread = std::make_unique<std::thread>(
        std::bind(&WhisperKit::TranscribeTask::Runtime::encode_decode_postproc_priv, this, timestamp)
    );
}


void Runtime::encode_decode_postproc_priv(float timestamp)
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

int Runtime::tflite_write_data_priv(int size, char* pcm_buffer) {
    audioinput->fill_pcmdata(size, pcm_buffer);

    return audioinput->get_curr_buf_time();
}
void Runtime::tflite_perfjson_priv() {

}

constexpr const uint64_t INPUT_BUFFER_SIZE = (8<<20);
constexpr const uint64_t STREAM_READ_SIZE = (512<<10);  // has to be larger than 128KB

#ifndef av_err2string
#define av_err2string(errnum) \
    av_make_error_string( \
        (char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), \
        AV_ERROR_MAX_STRING_SIZE, errnum \
    )
#endif
static int cbDecodeInterrupt(void *ctx)
{
    // return whether to stop the input stream or not
    AudioCodec *codec = (AudioCodec*)ctx;
    if(codec->is_running())
        return 0;
    else return 1;
}

//=========== AudioCodec =================
AudioCodec::AudioCodec()
{
    _format_context = nullptr;
    _freq           = 0;
    _channel        = 0;
    _io_context     = nullptr;
    _io_buffer      = nullptr;
    _codec_context  = nullptr;
    _codec          = nullptr;
    _src_frame      = nullptr;
    _dst_frame      = nullptr;
    _sample_format  = AV_SAMPLE_FMT_S16P;
    _is_wav_input   = false;
    _single_ch_datasize = 0;
}

AudioCodec::~AudioCodec()
{
    assert(!_format_context);
}

bool AudioCodec::open(string filename)
{
    int ret;
    AVCodecParameters *codec_par = nullptr;
    AVDictionary **opts = nullptr;
    AVDictionary *format_opts = nullptr;

    // allocating Format I/O context
    _format_context = avformat_alloc_context();
    _src_frame = av_frame_alloc();
    _dst_frame = av_frame_alloc();

    if (!_format_context)
    {
        LOGI("alloc format context failed\n");
        return false;
    }

    av_dict_set(&format_opts, "scan_all_pmts", "1", AV_DICT_DONT_OVERWRITE);

    _format_context->interrupt_callback.callback = cbDecodeInterrupt;
    _format_context->interrupt_callback.opaque = this;
    _format_context->max_analyze_duration = 1024000;
    _is_running = true;
    
    AVInputFormat *input_format = nullptr;

    ret = avformat_open_input(
        &_format_context, filename.c_str(), input_format, &format_opts
    );
    if (ret < 0)
    {
        LOGE("avformat_open_input Error: %s\n", av_err2string(ret));
        return false;
    }
    if (filename.find(".wav") != string::npos
        || filename.find(".wave") != string::npos)
    {
        _is_wav_input = true;
    }

    opts = (AVDictionary**)av_calloc(_format_context->nb_streams, sizeof(*opts));
    avformat_find_stream_info(_format_context, opts);
    _duration = (_format_context->duration)/1000;

    for (unsigned int i = 0; i < _format_context->nb_streams; i++)
    {
        codec_par = _format_context->streams[i]->codecpar;
        if(codec_par->codec_type == AVMEDIA_TYPE_AUDIO)
            break;
    }
    assert(codec_par != nullptr);
    _freq = codec_par->sample_rate;
    _channel = codec_par->ch_layout.nb_channels;
    _sample_format  = (AVSampleFormat)codec_par->format;

    if (!_is_wav_input)
    {
        _codec = (AVCodec*)avcodec_find_decoder(codec_par->codec_id);
        if (!_codec)
        {
            LOGE("avcodec_find_decoder failed\n");
            return false;
        }
        _codec_name = string(avcodec_get_name(codec_par->codec_id));
        
        LOGI("Audio Codec: %s\n", _codec_name.c_str());
        if(_codec_context != nullptr){
            avcodec_free_context(&_codec_context);
        }
        _codec_context = avcodec_alloc_context3(_codec);

        avcodec_parameters_to_context(_codec_context, codec_par);

        if (avcodec_open2(_codec_context, _codec, opts) < 0) {
            LOGE("Could not open audio codec..\n");
            return false;
        }
        _channel = 1; // we'll convert to mono, S16LE format

        _swr = swr_alloc();
        av_opt_set_chlayout(
            _swr, "in_chlayout", &_codec_context->ch_layout, 0
        );
        av_opt_set_int(
            _swr, "in_sample_rate", _codec_context->sample_rate, 0
        );
        av_opt_set_sample_fmt(
            _swr, "in_sample_fmt", (AVSampleFormat)codec_par->format, 0
        );

        _dst_frame->sample_rate = _codec_context->sample_rate;
        _dst_frame->format = AV_SAMPLE_FMT_S16;
        _dst_frame->ch_layout = AV_CHANNEL_LAYOUT_MONO;

        av_opt_set_chlayout(
            _swr, "out_chlayout", &_dst_frame->ch_layout,  0
        );
        av_opt_set_int(
            _swr, "out_sample_rate", _dst_frame->sample_rate, 0
        );
        av_opt_set_sample_fmt(
            _swr, "out_sample_fmt", (AVSampleFormat)_dst_frame->format,  0
        );
        ret = swr_init(_swr);
        if (ret < 0) {
            LOGE("Error in swr_init: %s\n", av_err2string(ret));
            return false;
        }
    }
    
    av_dump_format(_format_context, 0, nullptr, false);
    av_free(opts);

    return true;
}


void AudioCodec::close()
{
    if(_codec_context){
        av_free(_codec_context);
        _codec_context = nullptr;
    }
    if(_dst_frame){
        av_frame_unref(_dst_frame);
        av_frame_free(&_dst_frame);
        _dst_frame = nullptr;
        av_frame_unref(_src_frame);
        av_frame_free(&_src_frame);
        _src_frame = nullptr;
    }
    if (_swr) {
        swr_free(&_swr);
        _swr = nullptr;
    }
    if (_io_context) {
        avio_flush(_io_context);
        av_freep(&_io_context->buffer); // note that it is referencing m_pIOBuffer
        av_freep(&_io_context);
        _io_buffer = nullptr;
    }
    if (_format_context)
    {
        avformat_close_input(&_format_context);
        avformat_free_context(_format_context);
        _format_context = nullptr;
    }
    _is_running = false;
}

int AudioCodec::decode_pcm()
{
    int retry = 0;
    AVPacket packet;

    int ret = av_read_frame(_format_context, &packet);
    if( ret < 0)
    {
        return ret;
    }

    if (_is_wav_input && packet.size > 0){
        _dst_frame->data[0] = packet.data;  
        _single_ch_datasize = packet.size;
        return 0;          
    }

    if(packet.size > 0) {
        ret = avcodec_send_packet(_codec_context, &packet);
        if (ret < 0) {
            LOGE("Error sending a packet: %s\n", av_err2string(ret));
            return ret;
        }
    }

    av_frame_unref(_src_frame);
    ret = avcodec_receive_frame(_codec_context, _src_frame);
    if (ret == AVERROR(EAGAIN)) {
        return ret;
    } else if (ret == AVERROR_EOF) {
        return ret;
    } else if (ret < 0) {
        LOGE("Error during decoding: %s\n", av_err2string(ret));
        return ret;
    }

    if(_sample_format != AV_SAMPLE_FMT_S16P){
        av_frame_unref(_dst_frame);

        _dst_frame->sample_rate = _codec_context->sample_rate;
        _dst_frame->format = AV_SAMPLE_FMT_S16;
        _dst_frame->ch_layout = AV_CHANNEL_LAYOUT_MONO;

        ret = swr_convert_frame(_swr, _dst_frame, _src_frame);
        if (ret < 0) {
            LOGE("Error in swr_convert_frame: %s\n", av_err2string(ret));
            return -1;
        }
    } else 
        av_frame_move_ref(_dst_frame, _src_frame);

    _single_ch_datasize = _dst_frame->nb_samples 
        * av_get_bytes_per_sample((AVSampleFormat)_dst_frame->format);

    return 0;
}

int AudioCodec::get_format() const { 
    return _dst_frame->format; 
}

}

#ifdef av_err2string
#undef av_err2string
#endif

using namespace WhisperKit::TranscribeTask;

TranscribeTask::TranscribeTask(const whisperkit_configuration_t& config) {
    this->config = config;

    audio_codec = make_unique<AudioCodec>();
    argsjson = make_unique<nlohmann::json>();

    runtime = std::make_unique<Runtime>(config);
    runtime->tflite_init_priv();
    printf("tflite initialized\n");
}

TranscribeTask::~TranscribeTask() {
    runtime->tflite_close_priv();
}

void TranscribeTask::textOutputProc() {

    text_out_thread = make_unique<thread>([this](){
    unique_lock<mutex> ulock(runtime->messenger->_mutex);

    runtime->messenger->_cond_var.wait(ulock, [this]
    {
        runtime->messenger->print();
        return !runtime->messenger->_running;
    });
    });

}

void TranscribeTask::transcribe(const char* audio_file, whisperkit_transcription_result_t* transcription_result) {


    if( !audio_codec->open(audio_file) ){ 
        LOGE("Error opening audio file: %s\n", audio_file);
        throw std::runtime_error("Error opening audio file");
    }

    runtime->tflite_init_audioinput(audio_codec.get(), audio_file);


    printf("TranscribeTask::transcribe: __LINE__: %d\n", __LINE__);
    textOutputProc();
    printf("TranscribeTask::transcribe: __LINE__: %d\n", __LINE__);

    int pcm_secs = 0, ret = 0;
    while(true) {
        if (ret != AVERROR_EOF) {
            ret = audio_codec->decode_pcm();
            if(ret < 0 || audio_codec->get_datasize() == 0) {
                usleep(100000); // 100ms
                continue;
            }

            pcm_secs = tflite_write_data(
                (char*)audio_codec->get_frame()->extended_data[0], 
                audio_codec->get_datasize()
            );
            
            if (pcm_secs < 30) {
                continue;
            }
        }
        if(tflite_loop() < 0) {
            break;
        }
    } 
        printf("TranscribeTask::transcribe: __LINE__: %d\n", __LINE__);

    tflite_close();
    text_out_thread->join();
    text_out_thread.reset();
    printf("TranscribeTask::transcribe: __LINE__: %d\n", __LINE__);

    auto perfjson = tflite_perfjson();
    printf("TranscribeTask::transcribe: __LINE__: %d\n", __LINE__);

    LOGI("\nModel latencies:\n"
        "  Audio Input: %d inferences,\t median:%.2f ms\n"
        "  Melspectro: %d inferences,\t median:%.2f ms\n"
        "  Encoder: %d inferences,\t median:%.2f ms\n"
        "  Decoder: %d inferences,\t median:%.2f ms\n"
        "  Postproc: %d inferences,\t median:%.2f ms\n"
        "=========================\n"
        "Total Duration:\t %.3f ms\n\n",
        (int)(*perfjson)["audioinput"]["inf"], 
        (float)(*perfjson)["audioinput"]["med"], 
        (int)(*perfjson)["melspectro"]["inf"], 
        (float)(*perfjson)["melspectro"]["med"], 
        (int)(*perfjson)["encoder"]["inf"], 
        (float)(*perfjson)["encoder"]["med"], 
        (int)(*perfjson)["decoder"]["inf"], 
        (float)(*perfjson)["decoder"]["med"],
        (int)(*perfjson)["postproc"]["inf"], 
        (float)(*perfjson)["postproc"]["med"],
        (float)(*perfjson)["duration"]
    );

    printf("TranscribeTask::transcribe: __LINE__: %d\n", __LINE__);

    /*
    std::string modelPath = config.get_audio_encoder();
    std::string modelSize = "default"; // delete..
    if (config.get_verbose()) {
        auto testjson = get_test_json(
            modelPath.c_str(), 
            modelSize.c_str(), 
            (float)(*perfjson)["duration"]/1000
        ); 

        ofstream out_file;
        struct stat sb;

        if (stat(config.get_lib_dir().c_str(), &sb) != 0){
            mkdir(config.get_lib_dir().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        }
        out_file.open(config.get_lib_dir() + "/output.json");
        out_file << testjson->dump() << endl;
        out_file.close();
    }
    */

}

