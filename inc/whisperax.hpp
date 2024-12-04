//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#include <nlohmann/json.hpp>
#include <tflite_msg.hpp>


#if (defined(QNN_DELEGATE) || defined(GPU_DELEGATE)) 
// for Android QNN or GPU delegate
#define TFLITE_ROOT_PATH    "/sdcard/argmax/tflite"
#define DEFAULT_LIB_DIR     "/data/local/tmp/lib"
#define DEFAULT_CACHE_DIR   "/data/local/tmp/cache"
#else
#define TFLITE_ROOT_PATH    "."
#define DEFAULT_LIB_DIR     "./lib"
#define DEFAULT_CACHE_DIR   "./cache"
#endif

using json = nlohmann::json;

extern shared_ptr<TFLiteMessenger> messenger;

extern int tflite_init(string argstr);
extern int tflite_loop();
extern int tflite_close();
extern unique_ptr<json> tflite_perfjson();
extern int tflite_write_data(char* pcm_buffer, int size);

extern unique_ptr<json> get_test_json(
    const char* audiofile, 
    const char* model_size,
    float duration
); 