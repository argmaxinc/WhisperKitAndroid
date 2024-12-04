#include <whisperax.hpp>
#include "audio_codec.hpp"

#define av_err2string(errnum) \
    av_make_error_string( \
        (char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), \
        AV_ERROR_MAX_STRING_SIZE, errnum \
    )

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
