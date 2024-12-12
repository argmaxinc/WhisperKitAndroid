#!/bin/bash
# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

# only supposed to be run for CI purpose
# echo "  ${0} <act> <plat>"
# echo "      <act>: build, clean, test, rebuild-env"
# echo "      [plat]: qnn, gpu, linux, or all"

ACT=$1
PLAT=$2
CURRENT_DIR="$(dirname "$(realpath "$0")")"
SOURCE_DIR="$CURRENT_DIR/.."
IMAGE_NAME="android-ndk-qnn-tensorflow-image"
CONTAINER_NAME="axie_tflite"

function docker_start() {
    make ci-env
    sleep 1

    CONTAINER_ID="$(docker ps -aqf "name=$CONTAINER_NAME")"
    while [ "$CONTAINER_ID" == "" ]; do
        sleep 1
        CONTAINER_ID="$(docker ps -aqf "name=$CONTAINER_NAME")"
    done

    IS_RUNNING="$(docker inspect -f {{.State.Running}} $CONTAINER_ID)"
    while [ "$IS_RUNNING" == "false" ]; do
        sleep 1
        IS_RUNNING="$(docker inspect -f {{.State.Running}} $CONTAINER_ID)"
    done
}

case $ACT in
    "clean")
    if [ "$PLAT" = "all" ]; then
        cd $SOURCE_DIR; make clean all
    elif [ "$PLAT" = "linux" ]; then
        cd $SOURCE_DIR; make clean linux
    elif [ "$PLAT" = "gpu" ]; then
        cd $SOURCE_DIR; make clean gpu
    else
        cd $SOURCE_DIR; make clean
    fi
    ;;

    "build")
    docker_start

    if [ "$PLAT" = "linux" ]; then
        docker exec -i $CONTAINER_NAME /bin/bash -c "cd /src/AXIE; make build linux"
    elif [ "$PLAT" = "gpu" ]; then
        docker exec -i $CONTAINER_NAME /bin/bash -c "cd /src/AXIE; make build gpu"
    else
        docker exec -i $CONTAINER_NAME /bin/bash -c "cd /src/AXIE; make build"
    fi
    ;;

    "test")
    docker_start
    make download-models

    if [ "$PLAT" = "linux" ]; then
        docker exec -i $CONTAINER_NAME /bin/bash -c "cd /src/AXIE; make test linux"
    else
        make adb-push
        make test
    fi
    ;;

    "rebuild-env")
    make rebuild-env
esac
