#include "tensor.h"
#include "random.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

tensor_t *allocTensor(uint8_t dim, uint32_t *shape) {
  tensor_t *tensor = (tensor_t *)malloc(sizeof(tensor_t));
  if (!tensor) {
    perror("malloc");
    return NULL;
  }

  tensor->dim = dim;

  tensor->shape = (uint32_t *)malloc(dim * sizeof(uint32_t));
  if (!tensor->shape) {
    perror("malloc");
    free(tensor);
    return NULL;
  }

  tensor->stride = (uint32_t *)malloc(dim * sizeof(uint32_t));
  if (!tensor->stride) {
    perror("malloc");
    free(tensor->shape);
    free(tensor);
    return NULL;
  }

  uint32_t len = 1;
  for (int i = 0; i < dim; i++) {
    len *= shape[i];
    tensor->shape[i] = shape[i];
  }

  tensor->stride[dim - 1] = 1;
  for (int i = dim - 2; i >= 0; i--) {
    tensor->stride[i] = tensor->stride[i + 1] * tensor->shape[i + 1];
  }

  tensor->len = len;

  return tensor;
}

tensor_t *getTensor(uint8_t dim, uint32_t *shape) {
  tensor_t *tensor = allocTensor(dim, shape);

  tensor->storage = (storage_t *)malloc(sizeof(storage_t));
  if (!tensor->storage) {
    perror("malloc");
    free(tensor->stride);
    free(tensor->shape);
    free(tensor);
    return NULL;
  }

  tensor->storage->refCount = 1;

  tensor->storage->raw = (float *)malloc(tensor->len * sizeof(float));
  if (!tensor->storage->raw) {
    perror("malloc");
    free(tensor->storage);
    free(tensor->stride);
    free(tensor->shape);
    free(tensor);
    return NULL;
  }

  tensor->data = tensor->storage->raw;

  return tensor;
}

tensor_t *getView(tensor_t *tensor, uint32_t id) {
  tensor_t *slice = allocTensor(tensor->dim - 1, tensor->shape + 1);

  tensor->storage->refCount++;
  slice->storage = tensor->storage;
  slice->data = tensor->data + (id * tensor->stride[0]);

  return slice;
}

int fillTensor(tensor_t *tensor, float value) {
  for (uint32_t i = 0; i < tensor->len; i++) {
    tensor->data[i] = value;
  }
  return 0;
}

int fillGaussTensor(tensor_t *tensor) {
  for (uint32_t i = 0; i < tensor->len; i++) {
    tensor->data[i] = randGauss();
  }
  return 0;
}

int copyTensor(tensor_t *dest, tensor_t *origin) {
  if (!dest | !origin) {
    return 1;
  }
  if (dest->dim != origin->dim) {
    return 1;
  }
  if (memcmp(dest->shape, origin->shape, dest->dim)) {
    return 1;
  }

  memcpy(dest->data, origin->data, dest->len);

  return 0;
}

void freeTensor(tensor_t *tensor) {
  if (!tensor) {
    return;
  }
  tensor->storage->refCount--;
  if (tensor->storage->refCount == 0) {
    free(tensor->storage->raw);
    free(tensor->storage);
  }
  free(tensor->shape);
  free(tensor->stride);
  free(tensor);
  tensor = NULL;
}

int multiply2dTensor(tensor_t *prod, tensor_t *a, tensor_t *b) {
  if ((prod->dim !=2) | (a->dim != 2) | (b->dim != 2) |
  (prod->shape[0] != a->shape[0]) | (prod->shape[1] != b->shape[1]) |
  (a->shape[1] != b->shape[0])) {
    fprintf(stderr, "multiply2dTensor: dimention missmatch\n");
    return 1;
  }

  fillTensor(prod, 0);

  for (uint32_t i = 0; i < prod->shape[0]; i++) {
    for (uint32_t k = 0; k < a->shape[1]; k++) {
      for (uint32_t j = 0; j < prod->shape[1]; j++) {
        *getValue(prod, i, j) += *getValue(a, i, k) * *getValue(b, k, j);
      }
    }
  }

  return 0;
}

int addTensor(tensor_t *out, tensor_t *in) {
  if ((!out) | (!in) | (out->dim != in->dim) | (memcmp(out->shape, in->shape, out->dim * sizeof(uint32_t)))) {
    fprintf(stderr, "addTensor: dimention missmatch\n");
    return 1;
  }

  for (uint32_t i = 0; i < out->len; i++) {
    out->data[i] += in->data[i];
  }

  return 0;
}

int reluTensor(tensor_t *out, tensor_t *in) {
  if ((!out) | (!in) | (out->dim != in->dim) | (memcmp(out->shape, in->shape, out->dim * sizeof(uint32_t)))) {
    fprintf(stderr, "reluTensor: dimention missmatch\n");
    return 1;
  }

  for (uint32_t i = 0; i < out->len; i++) {
    out->data[i] = (in->data[i] > 0) ? in->data[i] : 0;
  }

  return 0;
}

int softmaxTensor(tensor_t *out, tensor_t *in) {
  if ((!out) | (!in) | (out->dim != in->dim) | (memcmp(out->shape, in->shape, out->dim * sizeof(uint32_t)))) {
    fprintf(stderr, "softmaxTensor: dimention missmatch\n");
    return 1;
  }

  double *buffer = malloc(sizeof(double) * out->len);
  double denominator = 0;

  for (uint32_t i = 0; i < out->len; i++) {
    out->data[i] = exp((double)in->data[i]);
    denominator += out->data[i];
  }

  for (uint32_t i = 0; i < out->len; i++) {
    out->data[i] = (float)(out->data[i] / denominator);
  }

  free(buffer);
  return 0;
}

float crossEntropyLoss(tensor_t *predicted, uint8_t real) {
  double p = (predicted->data[real] > 1e-7) ? predicted->data[real] : 1e-7;
  return (float)(-1 * log(p));
}

tensor_t *transpose2dTensor(tensor_t *tensor) {
  if (!tensor) {
    return NULL;
  }
  tensor_t *tensorT = allocTensor(tensor->dim, tensor->shape);
  if (!tensorT) {
    return NULL;
  }

  tensorT->storage = tensor->storage;
  tensorT->data = tensor->data;
  tensor->storage->refCount++;

  tensorT->shape[0] = tensor->shape[1];
  tensorT->shape[1] = tensor->shape[0];
  tensorT->stride[0] = tensor->stride[1];
  tensorT->stride[1] = tensor->stride[0];


  return tensorT;
}

void display2dTensor(tensor_t *tensor) {
  for (uint32_t line = 0; line < tensor->shape[0]; line++) {
    for (uint32_t column = 0; column < tensor->shape[1]; column++) {
      printf("%5.2f ", *getValue(tensor, line, column));
    }
    printf("\n");
  }
  printf("\n");
}
