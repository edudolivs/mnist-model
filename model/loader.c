#include "tensor.h"
#include "loader.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int readNumDimention(uint8_t *dim, FILE *file) {
  uint8_t magicNum[4];

  size_t elemRead = fread(magicNum, sizeof(uint8_t), 4, file);
  if (elemRead != 4) {
    fprintf(stderr, "fread: error\n");
    return 1;
  }

  *dim = magicNum[3];

  return 0;
}

void swapBytes(uint32_t *val) {
  *val = (((*val) << 24) | (((*val) << 8) & 0x00FF0000) | (((*val) >> 8) & 0x0000FF00) | ((*val) >> 24));
}

int readSizeDimention(uint32_t *shape, FILE *file, uint8_t dim) {
  size_t elemRead = fread(shape, sizeof(uint32_t), dim, file);
  if (elemRead != dim) {
    fprintf(stderr, "fread: error\n");
    return 1;
  }

  for (int i = 0; i < dim; i++) {
    swapBytes(shape + i);
  }

  return 0;
}

int readData(tensor_t *tensor, FILE *file) {
  uint8_t *buffer = (uint8_t *)malloc(sizeof(uint8_t) * tensor->len);

  size_t elemRead = fread(buffer, 1, tensor->len, file);
  if (elemRead != tensor->len) {
    fprintf(stderr, "fread: error\n");
    return 1;
  }

  if (tensor->dim == 3) {
    for (uint32_t i = 0; i < tensor->len; i++) {
      tensor->data[i] = (float)buffer[i] / 255.0;
    }
  } else {
    for (uint32_t i = 0; i < tensor->len; i++) {
      tensor->data[i] = (float)buffer[i];
    }
  }

  free(buffer);
  buffer = NULL;
  return 0;
}

tensor_t *loadIdx(char *filePath) {
  FILE *file = fopen(filePath, "rb");
  uint8_t dim;

  readNumDimention(&dim, file);

  uint32_t shape[dim];

  readSizeDimention(shape, file, dim);

  tensor_t *tensor = getTensor(dim, shape);

  readData(tensor, file);

  return tensor;
}

void displayImage(tensor_t *tensor) {
  for (uint32_t line = 0; line < tensor->shape[0]; line++) {
    for (uint32_t column = 0; column < tensor->shape[1]; column++) {
      if (*(getValue(tensor, line, column)) < 0.5) {
        printf(". ");
      } else {
        printf("0 ");
      }
    }
    printf("\n");
  }
  printf("\n");
}
