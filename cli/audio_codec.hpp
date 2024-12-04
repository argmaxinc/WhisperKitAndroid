#pragma once

#include <iostream>
#include <string>

extern "C"
{
#include <libavformat/avformat.h>
#include <libavutil/time.h>
#include <libavutil/file.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>

#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

#define INPUT_BUFFER_SIZE   (8<<20)
#define STREAM_READ_SIZE    (512<<10)           // has to be larger than 128KB

using namespace std;

class AudioCodec
{
public:
    AudioCodec();
    ~AudioCodec();

    bool    open(string filename);
    void    close();

    int             get_samplerate()    { return _freq; }
    int             get_channel()       { return _channel; }
    int             get_format()        { return _dst_frame->format; }
    AVFrame*        get_frame()         { return _dst_frame; }
    int64_t         get_duration_ms()   { return _duration; }
    bool            is_running()        { return _is_running; }
    int             get_datasize()      { return _single_ch_datasize; }
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
