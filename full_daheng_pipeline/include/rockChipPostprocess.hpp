#include "../include/rknn_api.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

void dump_tensor_attr(rknn_tensor_attr* attr);

unsigned char* load_model(const char* filename, int* model_size);
