#ifndef TENSOR
#define TENSOR

#include <stdint.h>
#include <time.h>

typedef struct {
  uint8_t numDimentions;
  uint32_t *sizeDimentions;
  double *data;
} tensor_t;

tensor_t *getNullTensor();

void freeTensor(tensor_t *tensor);

#endif
