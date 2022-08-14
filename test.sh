#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="4g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create vicorob_uresnet-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm -it \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v vicorob_uresnet-output-$VOLUME_SUFFIX:/output/ \
        vicorob_uresnet

docker run --rm \
        -v vicorob_uresnet-output-$VOLUME_SUFFIX:/output/ \
        python:3.9-slim cat /output/results.json | python3 -m json.tool

docker run --rm \
        -v vicorob_uresnet-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        python:3.9-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm vicorob_uresnet-output-$VOLUME_SUFFIX
