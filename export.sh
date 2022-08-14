#!/usr/bin/env bash

./build.sh

docker save vicorob_uresnet | gzip -c > vicorob_uresnet.tar.gz
