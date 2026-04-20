#include "tensor.h"
#include "loader.h"
#include "random.h"
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

  for (uint32_t i = 0; i < dim; i++) {
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

lImages_t *getLabeledImages(char *imagesPath, char *labelsPath) {
  lImages_t *lImages = (lImages_t *)malloc(sizeof(lImages_t));
  if (!lImages) {
    perror("malloc");
    return NULL;
  }

  lImages->images = loadIdx(imagesPath);
  lImages->labels = loadIdx(labelsPath);

  return lImages;
}

shuffler_t *getShuffler(lImages_t *lImages) {
  shuffler_t *shuffler = (shuffler_t *)malloc(sizeof(shuffler_t));
  if (!shuffler) {
    perror("malloc");
    return NULL;
  }

  shuffler->indexes = getIndexesArray(lImages->images);
  shuffler->views = getViewArray(lImages->images);
  shuffler->sLabels = (float *)malloc(sizeof(float) * lImages->labels->len);
  shuffler->lImages = lImages;

  return shuffler;
}

tensor_t *getViewArray(tensor_t *tensor) {
  tensor_t *array = (tensor_t *)malloc(sizeof(tensor_t) * tensor->shape[0]);
  if (!array) {
    perror("malloc");
    return NULL;
  }

  for (uint32_t i = 0; i < tensor->shape[0]; i++) {
    initView(array + i, tensor, i);
  }

  return array;
}

uint32_t *getIndexesArray(tensor_t *tensor) {
  uint32_t *indexes = (uint32_t *)malloc(sizeof(uint32_t) * tensor->shape[0]);
  for (uint32_t i = 0; i < tensor->shape[0]; i++) {
    indexes[i] = i;
  }

  return indexes;
}

int shuffleData(shuffler_t *shuffler) {
  shuffleArray(shuffler->indexes, shuffler->lImages->labels->len);
  for (uint32_t i = 0; i < shuffler->lImages->labels->len; i++) {
    (shuffler->views + i)->data = shuffler->lImages->images->data + (shuffler->lImages->images->stride[0] * shuffler->indexes[i]);
    shuffler->sLabels[i] = shuffler->lImages->labels->data[shuffler->indexes[i]];
  }

  return 0;
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
