#include "tensor.h"
#include "loader.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

bool readNumDimention(tensor_t *tensor, FILE *file) {
  uint8_t magicNum[4];

  size_t bytesRead = fread(magicNum, 1, 4, file);

  tensor->numDimentions = magicNum[3];

  return 0;
}

uint32_t swapBytes(uint32_t val) {
  return ((val << 24) | ((val << 8) & 0x00FF0000) | ((val >> 8) & 0x0000FF00) | (val >> 24));
}

bool readSizeDimention(tensor_t *tensor, FILE *file) {
  tensor->sizeDimentions = (uint32_t *)malloc(tensor->numDimentions * sizeof(uint32_t));

  size_t bytesRead = fread(tensor->sizeDimentions, 4, tensor->numDimentions, file);

  for (int i = 0; i < tensor->numDimentions; i++) {
    tensor->sizeDimentions[i] = swapBytes(tensor->sizeDimentions[i]);
  }

  return 1;
}

bool readData(tensor_t *tensor, FILE *file) {
  int totalSize = 1;

  for (int i = 0; i < tensor->numDimentions; i++) {
    totalSize *= tensor->sizeDimentions[i];
  }

  uint8_t *buffer = (uint8_t *)malloc(totalSize * sizeof(uint8_t));

  tensor->data = (double *)malloc(sizeof(double) * totalSize);

  size_t bytesRead = fread(buffer, 1, totalSize, file);

  if (tensor->numDimentions == 3) {
    for (int i = 0; i < totalSize; i++) {
      tensor->data[i] = (double)buffer[i] / 255.0;
    }
  } else {
    for (int i = 0; i < totalSize; i++) {
      tensor->data[i] = (double)buffer[i];
    }
  }

  free(buffer);
  buffer = NULL;
  return 0;
}

bool loadIdx(tensor_t *tensor, char *filePath) {
  FILE *file = fopen(filePath, "rb");

  readNumDimention(tensor, file);

  readSizeDimention(tensor, file);

  readData(tensor, file);

  return 0;
}

void displayImage(tensor_t *tensor, int imageId) {
  for (int line = 0; line < tensor->sizeDimentions[1]; line++) {
    for (int column = 0; column < tensor->sizeDimentions[2]; column++) {
      if (getPixelValue(tensor, imageId, line, column) < 0.5) {
        printf(". ");
      } else {
        printf("0 ");
      }
    }
    printf("\n");
  }
  printf("\n");
}
