//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#include <string>
#include <memory>
#include <thread>
#include <condition_variable>
#include <mutex>

#ifdef ANDROID_JNI
#include <android/log.h>

#define TAG "com.argmax.whisperax"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#else

#define LOGI(...) fprintf(stdout, __VA_ARGS__)
#define LOGE(...) fprintf(stderr, __VA_ARGS__)

#endif

using namespace std;

class TFLiteMessenger{
public:
    TFLiteMessenger() {
        _msg = make_unique<string>();
    }
    mutex _mutex;
    condition_variable _cond_var;

    bool _running = false;
    float _timestamp = 0;
    unique_ptr<string> _msg;

    void print(){
        if (_msg->empty()) {
            return;
        }
        if (!_running) {
            LOGI("\nFinal Text: %s\n", _msg.get()->c_str());
        } else {
            LOGI("\nText: %s\n", _msg.get()->c_str());
        }
    };
};
