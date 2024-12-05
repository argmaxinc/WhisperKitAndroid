# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

SCRIPTS_DIR = ./scripts

# Define targets for each script
.PHONY: setup env clean rebuild-env download-models build adb-push adb-shell help

help:
	@echo "Available targets:"
	@echo "  setup             Checking dependencies and any setup for the host."
	@echo "  download-models   Download all models and files."
	@echo "  env               Builds and runs docker environment to build axie_tflite CLI."
	@echo "  rebuild-env       Rebuilds and runs docker environment."
	@echo "  build             Build the axie_tflite CLI. Run this inside development environment."
	@echo "  adb-push          Push axie_tflite CLI and other dependencies to the Android device. Run this on host."
	@echo "  adb-shell         Open an interactive ADB shell and setups environment. Run this on host."
	@echo "  clean             Clean up previous build, both Android and x86."


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

rebuild-env:
	@bash $(SCRIPTS_DIR)/dev_env.sh -r

clean:
	@bash $(SCRIPTS_DIR)/build.sh clean

build:
	@bash $(SCRIPTS_DIR)/build.sh

adb-push:
	@bash $(SCRIPTS_DIR)/adb_push.sh

adb-shell:
	@expect $(SCRIPTS_DIR)/adb_shell.sh
