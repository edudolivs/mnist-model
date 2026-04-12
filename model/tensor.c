#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>

tensor_t *getNullTensor() {
  tensor_t *tensor = (tensor_t *)malloc(sizeof(tensor_t));
  if (!tensor) {
    perror("malloc");
    return NULL;
  }
  tensor->sizeDimentions = NULL;
  tensor->data = NULL;
  return tensor;
}

void freeTensor(tensor_t *tensor) {
  if (!tensor) {
    return;
  }
  if (tensor->sizeDimentions) {
    free(tensor->sizeDimentions);
  }
  if (tensor->data) {
    free(tensor->data);
  }
  free(tensor);
  tensor = NULL;
}

