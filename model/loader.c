#include "tensor.h"
#include "loader.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uint8_t readNumDimention(FILE *file) {
  uint8_t magicNum[4];

  size_t bytesRead = fread(magicNum, 1, 4, file);

  return magicNum[3];
}

uint32_t swapBytes(uint32_t val) {
  return ((val << 24) | ((val << 8) & 0x00FF0000) | ((val >> 8) & 0x0000FF00) | (val >> 24));
}

bool readSizeDimention(uint32_t *shape, FILE *file, uint8_t dim) {
  size_t bytesRead = fread(shape, sizeof(uint32_t), dim, file);

  for (int i = 0; i < dim; i++) {
    shape[i] = swapBytes(shape[i]);
  }

  return shape;
}

bool readData(tensor_t *tensor, FILE *file) {
  uint8_t *buffer = (uint8_t *)malloc(sizeof(uint8_t) * tensor->len);

  size_t bytesRead = fread(buffer, 1, tensor->len, file);

  if (tensor->dim == 3) {
    for (int i = 0; i < tensor->len; i++) {
      tensor->data[i] = (float)buffer[i] / 255.0;
    }
  } else {
    for (int i = 0; i < tensor->len; i++) {
      tensor->data[i] = (float)buffer[i];
    }
  }

  free(buffer);
  buffer = NULL;
  return 0;
}

tensor_t *loadIdx(char *filePath) {
  FILE *file = fopen(filePath, "rb");

  uint8_t dim = readNumDimention(file);

  uint32_t shape[dim];

  readSizeDimention(shape, file, dim);

  tensor_t *tensor = getTensor(dim, shape);

  readData(tensor, file);

  return tensor;
}

void displayImage(tensor_t *tensor) {
  for (int line = 0; line < tensor->shape[0]; line++) {
    for (int column = 0; column < tensor->shape[1]; column++) {
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
