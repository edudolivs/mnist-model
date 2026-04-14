#include "tensor.h"
#include <stdint.h>
#include <math.h>

typedef struct {
  uint32_t sizeIn;
  uint32_t sizeOut;

  tensor_t *weights;
  tensor_t *biases;

  tensor_t *input;
  tensor_t *z;
  tensor_t *output;

  tensor_t *gWeights;
  tensor_t *gBiases;
} layer_t;

layer_t *getLayer(uint32_t sizeIn, uint32_t sizeOut);

void freeLayer(layer_t *layer);
