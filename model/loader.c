#include "loader.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

array_t *getArray() {
  array_t *array = (array_t *)malloc(sizeof(array_t));
  array->sizeDimentions = NULL;
  array->data = NULL;
  return array;
}

void freeArray(array_t *array) {
  if (!array) {
    return;
  }
  if (array->sizeDimentions) {
    free(array->sizeDimentions);
  }
  if (array->data) {
    free(array->data);
  }
  array = NULL;
}

bool readNumDimention(array_t *array, FILE *file) {
  uint8_t magicNum[4];

  size_t bytesRead = fread(magicNum, 1, 4, file);

  array->numDimentions = magicNum[3];

  if (array->numDimentions == 3) {
    array->category = IMAGE;
  } else {
    array->category = LABEL;
  }

  return 0;
}

uint32_t swapBytes(uint32_t val) {
  return ((val << 24) | ((val << 8) & 0x00FF0000) | ((val >> 8) & 0x0000FF00) | (val >> 24));
}

bool readSizeDimention(array_t *array, FILE *file) {
  array->sizeDimentions = (uint32_t *)malloc(array->numDimentions * sizeof(uint32_t));

  size_t bytesRead = fread(array->sizeDimentions, 4, array->numDimentions, file);

  for (int i = 0; i < array->numDimentions; i++) {
    array->sizeDimentions[i] = swapBytes(array->sizeDimentions[i]);
  }

  return 1;
}

bool readData(array_t *array, FILE *file) {
  int totalSize = 1;

  for (int i = 0; i < array->numDimentions; i++) {
    totalSize *= array->sizeDimentions[i];
  }

  uint8_t *buffer = (uint8_t *)malloc(totalSize * sizeof(uint8_t));

  array->data = (float *)malloc(sizeof(float) * totalSize);

  size_t bytesRead = fread(buffer, 1, totalSize, file);

  if (array->category == IMAGE) {
    for (int i = 0; i < totalSize; i++) {
      array->data[i] = (float)buffer[i] / 255.0;
    }
  } else {
    for (int i = 0; i < totalSize; i++) {
      array->data[i] = (float)buffer[i];
    }
  }

  return 0;
}

bool loadIdx(array_t *array, char *filePath) {
  FILE *file = fopen(filePath, "rb");

  readNumDimention(array, file);

  readSizeDimention(array, file);

  readData(array, file);

  return 0;
}

void displayImage(array_t *array, int imageId) {
  for (int line = 0; line < array->sizeDimentions[1]; line++) {
    for (int column = 0; column < array->sizeDimentions[2]; column++) {
      if (getPixelValue(array, imageId, line, column) < 0.5) {
        printf(". ");
      } else {
        printf("0 ");
      }
    }
    printf("\n");
  }
  printf("\n");
}
