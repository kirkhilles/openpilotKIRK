#ifndef IMGPROC_UTILS
#define IMGPROC_UTILS

#include <stdint.h>

#define NUM_SEGMENTS_X 8
#define NUM_SEGMENTS_Y 6

#define ROI_X_MIN 1
#define ROI_X_MAX 6
#define ROI_Y_MIN 2
#define ROI_Y_MAX 3

#define LM_THRESH 120
#define LM_PREC_THRESH 0.9 // 90 perc is blur

// only apply to QCOM
#define FULL_STRIDE_X 1280
#define FULL_STRIDE_Y 896

#define CONV_LOCAL_WORKSIZE 16

const int16_t lapl_conv_krnl[9] = {0, 1, 0,
                                  1, -4, 1,
                                  0, 1, 0};

void get_lapmap_one(int16_t *lap, uint16_t *res, int x_pitch, int y_pitch);
bool is_blur(uint16_t *lapmap);

#endif
