#!/bin/bash
# Profile kernels with ncu
cd "$(dirname "$0")/build"
ncu --set full ./supersonic
