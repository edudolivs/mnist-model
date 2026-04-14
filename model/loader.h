#ifndef LOADER
#define LOADER

#include "tensor.h"
#include <stdint.h>
#include <stdio.h>

int readNumDimention(uint8_t *dim, FILE *file);

void swapBytes(uint32_t *val);

int readSizeDimention(uint32_t *shape, FILE *file, uint8_t dim);

int readData(tensor_t *tensor, FILE *file);

tensor_t *loadIdx(char *filePath);

void displayImage(tensor_t *tensor);

#endif
