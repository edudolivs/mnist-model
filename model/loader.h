#ifndef LOADER
#define LOADER

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef enum {
  LABEL = 1,
  IMAGE = 3,
} category_t;

typedef struct {
  category_t category;
  uint8_t numDimentions;
  uint32_t *sizeDimentions;
  float *data;
} array_t;

array_t *getArray();

void freeArray(array_t *array);

static bool readNumDimention(array_t *array, FILE *file);

uint32_t swapBytes(uint32_t val);

static bool readSizeDimention(array_t *array, FILE *file);

static bool readData(array_t *array, FILE *file);

bool loadIdx(array_t *array, char *filePath);

static inline float getPixelValue(array_t *array, int imageId, int line, int column) {
  return array->data[array->sizeDimentions[1] * array->sizeDimentions[2] * imageId + array->sizeDimentions[2] * line + column];
}

void displayImage(array_t *array, int imageId);

#endif
