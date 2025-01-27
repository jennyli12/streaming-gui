# streaming-gui

Real-time plotting for PicoScopes, a microphone, and the Polar Verity Sense.

## Setup for Apple Silicon Macs

1) Set up a [Rosetta terminal](https://osxdaily.com/2020/11/18/how-run-homebrew-x86-terminal-apple-silicon-mac/). Download the Miniconda command-line installer (64-bit Intel) [here](https://www.anaconda.com/download/), and execute it in the Rosetta terminal with `bash`. You might want to rename the install folder to something like `miniconda-rosetta` if you also want this to coexist with a native conda installation. The following steps do not require the Rosetta terminal.
2) Create a new environment with `conda create --name pico` and activate it with `conda activate pico`. Install Python 3.9 with `conda install python=3.9`. Use `which python` and `which pip` to check you are running the correct installation.
3) Install PyAudio with `conda install pyaudio`.
4) Download PicoSDK 11.1.0 for Mac [here](https://www.picotech.com/downloads). 
5) Download the PicoSDK Python package [here](https://github.com/picotech/picosdk-python-wrappers). Run `pip install .` in the top-level directory. You can delete this repository if you'd like.
6) Install other requirements with `pip install PyQt5 pyqtgraph qasync bleak bitstring matplotlib`.
7) Run `export DYLD_LIBRARY_PATH=/Library/Frameworks/PicoSDK.framework/Libraries/libpicoipp/:/Library/Frameworks/PicoSDK.framework/Libraries/libps3000a:/Library/Frameworks/PicoSDK.framework/Libraries/libps4000` to correctly link to the driver packages. You can add this to your `.zshrc` or similar if you don't want to repeat it for every new terminal.
