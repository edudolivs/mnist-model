#ifndef TENSOR
#define TENSOR

#include <stdint.h>
#include <time.h>

typedef struct {
  float *raw;
  uint32_t refCount;
} storage_t;

typedef struct {
  uint32_t *shape;
  uint32_t *stride;
  storage_t *storage;
  float *data;
  uint32_t len;
  uint8_t dim;
} tensor_t;

tensor_t *allocTensor(uint8_t dim, uint32_t *shape);

tensor_t *getTensor(uint8_t dim, uint32_t *shape);

int initView(tensor_t *view, tensor_t *tensor, uint32_t id);

tensor_t *getView(tensor_t *tensor, uint32_t id);

static inline float *getValue(tensor_t *tensor, uint32_t line, uint32_t column) {
  return tensor->data + (tensor->stride[0] * line + tensor->stride[1] * column);
}

int fillTensor(tensor_t *tensor, float value);

int fillGaussTensor(tensor_t *tensor);

int copyTensor(tensor_t *dest, tensor_t *origin);

void freeTensor(tensor_t *tensor);

void freeViewArray(tensor_t *viewArray, uint32_t len);

int multiply2dTensor(tensor_t *prod, tensor_t *a, tensor_t *b);

int addTensor(tensor_t *out, tensor_t *in);

int reluTensor(tensor_t *out, tensor_t *in);

int softmaxTensor(tensor_t *out, tensor_t *in);

float crossEntropy(tensor_t *tensor_t, uint8_t label);

tensor_t *transpose2dTensor(tensor_t *tensor);

void display2dTensor(tensor_t *tensor);

#endif
