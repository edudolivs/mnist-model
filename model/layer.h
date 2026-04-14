#include "tensor.h"
#include <stdint.h>

typedef struct {
  uint32_t sizeIn;
  uint32_t sizeOut;

  tensor_t *weights;
  tensor_t *biases;

  tensor_t *in;
  tensor_t *z;
  tensor_t *out;

  tensor_t *dWeights;
  tensor_t *dBiases;
} layer_t;

layer_t *getLayer(uint32_t sizeIn, uint32_t sizeOut);

void freeLayer(layer_t *layer);

int insertInput(layer_t *layer, tensor_t *input);

int computeLayer(layer_t *layer);
