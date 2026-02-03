#!/bin/bash

cd "$(dirname "$0")"

case "$1" in
    -v)
        ncu --set full ./build/supersonic
        ;;
    -o)
        ncu --set full -o "build/${2:-profile}" ./build/supersonic
        echo "Output: $(pwd)/build/${2:-profile}.ncu-rep"
        ;;
    -f)
        ncu --set full --force-overwrite -o "build/${2:-profile}" ./build/supersonic
        echo "Output: $(pwd)/build/${2:-profile}.ncu-rep"
        ;;
    *)
        ncu --metrics gpu__time_duration.avg ./build/supersonic
        ;;
esac
