#include <fstream>
#include <unistd.h>
#include <sys/stat.h>
#include <audio_codec.hpp>
#include <whisperax.hpp>

using namespace std;

#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "?? UNKNOWN"
#endif

unique_ptr<thread> text_out_thread;
unique_ptr<AudioCodec> audio_codec;

void textOutputProc() {
    text_out_thread = make_unique<thread>([](){
        unique_lock<mutex> ulock(messenger->_mutex);

        messenger->_cond_var.wait(ulock, []
        {
            messenger->print();
            return !messenger->_running;
        });
    });
}

int main(int argc, char* argv[]) {
    LOGI("Compiled: %s %s\nGit commit: %s\nArgmax, Inc.\n\n", 
        __DATE__, __TIME__, GIT_COMMIT_HASH
    );
    if (argc < 3) {
        LOGE("Usage: %s <audio input> <tiny | base | small>\n", argv[0]);
        return -1;
    }

    bool debug = false;
    if (argc > 3 && string(argv[3]).find("debug") != string::npos)
        debug = true; 

    string root_path = filesystem::current_path();

    audio_codec = make_unique<AudioCodec>();
    if( !audio_codec->open(string(argv[1])) ){ 
        LOGE("Error opening audio file: %s\n", argv[1]);
        return -1;
    }

    auto argsjson = make_unique<json>();
    (*argsjson)["lib"] = DEFAULT_LIB_DIR;
    (*argsjson)["cache"] = DEFAULT_CACHE_DIR;
    (*argsjson)["size"] = argv[2];
    (*argsjson)["freq"] = audio_codec->get_samplerate();
    (*argsjson)["dur"] = audio_codec->get_duration_ms();
    (*argsjson)["ch"] = audio_codec->get_channel(); 
    (*argsjson)["fmt"] = audio_codec->get_format(); 
    (*argsjson)["root_path"] = root_path;
    (*argsjson)["debug"] = debug;

    tflite_init(argsjson->dump());

    textOutputProc();

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
    tflite_close();
    text_out_thread->join();
    text_out_thread.reset();

    auto perfjson = tflite_perfjson();

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

    if (debug){
        auto testjson = get_test_json(
            argv[1], argv[2], 
            (float)(*perfjson)["duration"]/1000
        ); 

        ofstream out_file;
        struct stat sb;

        if (stat(root_path.c_str(), &sb) != 0){
            mkdir(root_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        }
        out_file.open(root_path + "/output.json");
        out_file << testjson->dump() << endl;
        out_file.close();
    }
    return 0;
}