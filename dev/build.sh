#!/usr/bin/env bash

set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR/../"

# compile XGBoost

cd $PROJECT_DIR;
rm -rf xgboost_upstream;
git clone --recursive git@github.com:CodingCat/xgboost.git xgboost_upstream
git fetch --all
git checkout dist_fast_histogram
cd $PROJECT_DIR/xgboost_upstream/jvm-packages;

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    mvn package -DskipTests;
elif [[ "$OSTYPE" == "darwin"* ]]; then
    . dev/build-docker.sh
fi

# install locally
mvn install -DskipTests

# build
cd $PROJECT_DIR;
mvn package