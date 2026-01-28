#!/bin/bash

cd "$(dirname "$0")/build"

case "$1" in
    -v)
        ncu --set full ./supersonic
        ;;
    -o)
        ncu --set full -o "${2:-profile}" ./supersonic
        echo "Output: $(pwd)/${2:-profile}.ncu-rep"
        ;;
    -f)
        ncu --set full --force-overwrite -o "${2:-profile}" ./supersonic
        echo "Output: $(pwd)/${2:-profile}.ncu-rep"
        ;;
    *)
        ncu --metrics gpu__time_duration.avg ./supersonic
        ;;
esac
