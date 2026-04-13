#ifndef TENSOR
#define TENSOR

#include <stdint.h>
#include <time.h>
#include <stdbool.h>

typedef struct {
  uint32_t *shape;
  uint32_t *stride;
  float *data;
  uint32_t len;
  uint8_t dim;
  bool isOwner;
} tensor_t;

tensor_t *getTensor(uint8_t dim, uint32_t *shape);

tensor_t *getView(tensor_t *tensor, uint32_t id);

static inline float *getValue(tensor_t *tensor, uint32_t line, uint32_t column) {
  return tensor->data + (tensor->stride[0] * line + tensor->stride[1] * column);
}

bool fillTensor(tensor_t *tensor, float value);

bool fillGaussTensor(tensor_t *tensor);

bool copyTensor(tensor_t *dest, tensor_t *origin);

void freeTensor(tensor_t *tensor);

bool multiply2dTensor(tensor_t *prod, tensor_t *a, tensor_t *b);

bool addTensor(tensor_t *dest, tensor_t *origin);

tensor_t *transpose2dTensor(tensor_t *tensor);

void display2dTensor(tensor_t *tensor);

#endif
