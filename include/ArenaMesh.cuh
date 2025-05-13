#pragma once

#include "CudaCommon.cuh"

/*
Read in mesh with DataStream
Put in CUDA texture memory for fast read-only access across threads
Constant broad phase grid can contain reference to texture memory
*/