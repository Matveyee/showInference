#ifndef _RKNN_YOLOV6_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV6_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"
#include "yolov6.h"


// class rknn_app_context_t;



int init_post_process();
void deinit_post_process();
char *coco_cls_to_name(int cls_id);
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);

void deinitPostProcess();
#endif //_RKNN_YOLOV6_DEMO_POSTPROCESS_H_