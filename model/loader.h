#ifndef LOADER
#define LOADER

#include "tensor.h"
#include <stdint.h>
#include <stdio.h>

typedef struct {
  tensor_t *images;
  tensor_t *labels;
} lImages_t;

typedef struct {
  lImages_t *lImages;
  tensor_t *views;
  float *sLabels;
  uint32_t *indexes;
} shuffler_t;

int readNumDimention(uint8_t *dim, FILE *file);

void swapBytes(uint32_t *val);

int readSizeDimention(uint32_t *shape, FILE *file, uint8_t dim);

int readData(tensor_t *tensor, FILE *file);

tensor_t *loadIdx(char *filePath);

lImages_t *getLabeledImages(char *imagesPath, char *labelsPath);

shuffler_t *getShuffler(lImages_t *lImages);

tensor_t *getViewArray(tensor_t *tensor);

uint32_t *getIndexesArray(tensor_t *tensor);

int shuffleData(shuffler_t *shuffler);

void displayImage(tensor_t *tensor);

#endif
