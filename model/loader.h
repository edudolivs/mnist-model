#ifndef LOADER
#define LOADER

#include "tensor.h"
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
} tensor_t;

tensor_t *getTensor();

void freeTensor(tensor_t *tensor);

static bool readNumDimention(tensor_t *tensor, FILE *file);

uint32_t swapBytes(uint32_t val);

static bool readSizeDimention(tensor_t *tensor, FILE *file);

static bool readData(tensor_t *tensor, FILE *file);

bool loadIdx(tensor_t *tensor, char *filePath);

static inline float getPixelValue(tensor_t *tensor, int imageId, int line, int column) {
  return tensor->data[tensor->sizeDimentions[1] * tensor->sizeDimentions[2] * imageId + tensor->sizeDimentions[2] * line + column];
}

void displayImage(tensor_t *tensor, int imageId);

#endif
