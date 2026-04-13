#ifndef LOADER
#define LOADER

#include "tensor.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

uint8_t readNumDimention(FILE *file);

uint32_t swapBytes(uint32_t val);

bool readSizeDimention(uint32_t *shape, FILE *file, uint8_t dim);

bool readData(tensor_t *tensor, FILE *file);

tensor_t *loadIdx(char *filePath);

void displayImage(tensor_t *tensor);

#endif
