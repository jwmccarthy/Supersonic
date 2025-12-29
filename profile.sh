#!/bin/bash

cd "$(dirname "$0")/build"

if [[ "$1" == "-v" ]]; then
    ncu --set full ./supersonic
else
    ncu --metrics gpu__time_duration.avg ./supersonic
fi
