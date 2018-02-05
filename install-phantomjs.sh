#!/usr/bin/env bash

system=$(python -c "from sys import platform; print(platform)")
rm -rf phantomjs

if [[ $system == "darwin" ]]; then
    phantomjs="phantomjs-2.1.1-macosx"
elif [[ $system == "win32" ]]; then
    phantomjs="phantomjs-2.1.1-windows"
elif [[ $system == "linux"* ]]; then
    phantomjs="phantomjs"
else
    echo "
    Your OS does not support phantomjs static build.
    To install phantomjs from source, please visit http://phantomjs.org/download.html"
    exit
fi

if [[ $system == "linux"* ]]; then
    url=https://github.com/ariya/phantomjs/releases/download/2.1.3/$phantomjs
else
    url=https://bitbucket.org/ariya/phantomjs/downloads/$phantomjs.zip
fi

if type curl &>/dev/null; then
    curl -RLO $url
elif type wget &>/dev/null; then
    wget -N -nc $url
fi

if [[ $system == "linux"* ]]; then
    # Can't have file and directory w/ same name. Using temp name.
    mv $phantomjs temp
    mkdir -p phantomjs/bin
    mv temp phantomjs/bin
    mv phantomjs/bin/temp phantomjs/bin/$phantomjs
else
    unzip $phantomjs.zip
    rm $phantomjs.zip
    mv $phantomjs phantomjs
fi

