#include "tensor.h"
#include "random.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

tensor_t *getTensor(uint8_t dim, uint32_t *shape) {
  tensor_t *tensor = (tensor_t *)malloc(sizeof(tensor_t));
  if (!tensor) {
    perror("malloc");
    return NULL;
  }

  tensor->isOwner = true;
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

  tensor->data = (float *)malloc(len * sizeof(float));
  if (!tensor->data) {
    perror("malloc");
    free(tensor->stride);
    free(tensor->shape);
    free(tensor);
    return NULL;
  }

  return tensor;
}

tensor_t *getView(tensor_t *tensor, uint32_t id) {
  tensor_t *slice = getTensor(tensor->dim - 1, tensor->shape + 1);

  slice->isOwner = false;
  free(slice->data);
  slice->data = tensor->data + (id * tensor->stride[0]);

  return slice;
}

bool fillTensor(tensor_t *tensor, float value) {
  for (int i = 0; i < tensor->len; i++) {
    tensor->data[i] = value;
  }
  return 0;
}

bool fillGaussTensor(tensor_t *tensor) {
  for (int i = 0; i < tensor->len; i++) {
    tensor->data[i] = randGauss();
  }
  return 0;
}

bool copyTensor(tensor_t *dest, tensor_t *origin) {
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
  if (tensor->isOwner) {
    free(tensor->data);
  }
  free(tensor->shape);
  free(tensor);
  tensor = NULL;
}

bool multiply2dTensor(tensor_t *prod, tensor_t *a, tensor_t *b) {
  if ((prod->dim !=2) | (a->dim != 2) | (b->dim != 2) |
  (prod->shape[0] != a->shape[0]) | (prod->shape[1] != b->shape[1]) |
  (a->shape[1] != b->shape[0])) {
    fprintf(stderr, "multiply2dTensor: dimention missmatch.");
    return 1;
  }

  fillTensor(prod, 0);

  for (int i = 0; i < prod->shape[0]; i++) {
    for (int k = 0; k < a->shape[1]; k++) {
      for (int j = 0; j < prod->shape[1]; j++) {
        *getValue(prod, i, j) += *getValue(a, i, k) * *getValue(b, k, j);
      }
    }
  }

  return 0;
}

bool addTensor(tensor_t *dest, tensor_t *origin) {
  if ((dest->dim != origin->dim) | (memcmp(dest->shape, origin->shape, dest->dim * sizeof(uint32_t)))) {
    fprintf(stderr, "addTensor: dimention missmatch.");
    return 1;
  }

  for (int i = 0; i < dest->len; i++) {
    dest->data[i] += origin->data[i];
  }

  return 0;
}

tensor_t *transpose2dTensor(tensor_t *tensor) {
  if (!tensor) {
    return NULL;
  }
  tensor_t *tensorT = getTensor(tensor->dim, tensor->shape);
  if (!tensorT) {
    return NULL;
  }

  free(tensorT->data);
  tensorT->data = tensor->data;
  tensorT->isOwner = false;

  tensorT->shape[0] = tensor->shape[1];
  tensorT->shape[1] = tensor->shape[0];
  tensorT->stride[0] = tensor->stride[1];
  tensorT->stride[1] = tensor->stride[0];


  return tensorT;
}

void display2dTensor(tensor_t *tensor) {
  for (int line = 0; line < tensor->shape[0]; line++) {
    for (int column = 0; column < tensor->shape[1]; column++) {
      printf("%5.2f ", *getValue(tensor, line, column));
    }
    printf("\n");
  }
  printf("\n");
}
