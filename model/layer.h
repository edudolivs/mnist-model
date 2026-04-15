#ifndef NETWORK
#define NETWORK

#include "tensor.h"
#include <stdint.h>

typedef enum {
  RELU,
  SOFT,
} act_t;

typedef struct {
  uint32_t sizeIn;
  uint32_t sizeOut;
  act_t act;

  tensor_t *weights;
  tensor_t *biases;

  tensor_t *in;
  tensor_t *z;
  tensor_t *out;

  tensor_t *dWeights;
  tensor_t *dBiases;
} layer_t;

typedef struct {
  uint32_t sizeIn;

  uint32_t numLayers;
  uint32_t *sizeLayers;

  layer_t *layers;
} network_t;

layer_t *getLayer(uint32_t sizeIn, uint32_t sizeOut, act_t act);

network_t *getNetwork(uint32_t sizeIn, uint32_t numLayers, uint32_t *sizeLayers);

void freeLayer(layer_t *layer);

void freeNetwork(network_t *network);

int insertInput(layer_t *layer, tensor_t *input);

int computeLayer(layer_t *layer);

int computeNetwork(network_t *network, tensor_t *input);

#endif
