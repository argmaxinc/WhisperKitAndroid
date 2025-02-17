# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

SCRIPTS_DIR = ./scripts

# Define targets for each script
.PHONY: setup env ci-env clean rebuild-env download-models build test adb-push help

args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

help:
	@echo "Available targets:"
	@echo "  setup             		Checking dependencies and any setup for the host."
	@echo "  download-models   		Download all models and files."
	@echo "  env               		Builds and runs docker environment to build axie_tflite CLI."
	@echo "  ci-env            		Builds and runs docker environment for GitHub CI"
	@echo "  rebuild-env       		Clean and rebuilds and runs docker environment."
	@echo "  clean        	   		Clean WhisperKitAndroid build."
	@echo "    [all]       	   		Deep clean WhisperKitAndroid build, including external components"
	@echo "  build             		Build the axie_tflite CLI. **Run this inside development environment** "
	@echo "    [qnn|gpu|linux|jni] 	Build for each target: QNN or GPU for Android, or Linux"
	@echo "  adb-push          		Push axie_tflite CLI and other dependencies to the Android device. Run this on host."
	@echo "  test              		Builds and install test dependencies."


setup:
	@echo "Setting up environment..."
	@echo "Checking for Aria2 ..."
	@which aria2c > /dev/null || (echo "Error: Aria2 is not installed. Install using 'brew install aria2' or other methods from https://aria2.github.io/ and try again" && exit 1)
	@echo "Checking for docker ..."
	@which docker > /dev/null || (echo "Error: Docker is not installed. Install docker from https://www.docker.com/ and try again." && exit 1)
	@echo "Checking for adb ..."
	@which adb > /dev/null || (echo "Error: Android Debug Bridge (adb) is not installed. Install it using 'brew install --cask android-platform-tools' or by installing Android Studio and try again." && exit 1)
	@echo "Checking for expect ..."
	@which expect > /dev/null || (echo "Error: expect is not installed. Install using 'brew install expect' or other method from https://core.tcl-lang.org/expect/index and try again" && exit 1)
	@echo "Done 🚀"

download-models:
	@bash $(SCRIPTS_DIR)/download_models.sh

env:
	@bash $(SCRIPTS_DIR)/dev_env.sh 

ci-env:
	@bash $(SCRIPTS_DIR)/dev_env.sh -c

rebuild-env:
	@bash $(SCRIPTS_DIR)/dev_env.sh -rf

clean:
	@bash $(SCRIPTS_DIR)/build.sh clean $(call args,) 

build:
	@bash $(SCRIPTS_DIR)/build.sh $(call args,)

test:
	@bash $(SCRIPTS_DIR)/build_test.sh $(call args,)

adb-push:
	@bash $(SCRIPTS_DIR)/adb_push.sh $(call args,)

all:	# do nothing - sub target of clean
	@echo ""

linux:	# do nothing - sub target of build/test
	@echo ""

qnn:	# do nothing - sub target of build/test
	@echo ""

gpu:	# do nothing - sub target of build/test
	@echo ""

jni:	# do nothing - sub target of build/test
	@echo ""

forced:	# do nothing - sub target of adb-push
	@echo ""