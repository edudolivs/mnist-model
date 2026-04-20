#ifndef NETWORK
#define NETWORK

#include "tensor.h"
#include "loader.h"
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

  float learningRate;
  uint32_t batchSize;

  layer_t *layers;
  float **buffers;
} network_t;

int initLayer(layer_t *layer, uint32_t sizeIn, uint32_t sizeOut, act_t act);

layer_t *getLayer(uint32_t sizeIn, uint32_t sizeOut, act_t act);

network_t *getNetwork(uint32_t sizeIn, uint32_t numLayers, uint32_t *sizeLayers, float learningRate, uint32_t batchSize);

void freeLayer(layer_t *layer);

void freeNetwork(network_t *network);

int zeroNetworkDerivative(network_t *network);

int insertInput(layer_t *layer, tensor_t *input);

int computeLayer(layer_t *layer);

int computeNetwork(network_t *network, tensor_t *input);

int accDeriveZ(layer_t *layer, float *dOut);

int accDeriveBiases(layer_t *layer, float *dOut);

int accDeriveWeights(layer_t *layer, float *dOut);

int accDeriveWO(layer_t *layer, float *dIn, float *dOut);

int accDeriveNetwork(network_t *network, uint8_t label);

int updateNetwork(network_t *network);

int trainEpoch(network_t *network, shuffler_t *shuffler);

#endif
