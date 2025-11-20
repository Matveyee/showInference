#include "rknn_api.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


int64_t getCurrentTimeUs();

int rknn_GetTopN(float* pfProb, float* pfMaxProb, uint32_t* pMaxClass, uint32_t outputCount, uint32_t topNum);

void dump_tensor_attr(rknn_tensor_attr* attr);

unsigned char* load_model(const char* filename, int* model_size);