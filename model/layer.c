#include "layer.h"
#include "tensor.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

int initLayer(layer_t *layer, uint32_t sizeIn, uint32_t sizeOut, act_t act) {
  uint32_t shapeOut[2] = {sizeOut, 1};
  layer->z = getTensor(2, shapeOut);
  if (!layer->z) {
    perror("malloc");
    return 1;
  }

  layer->out = getTensor(2, shapeOut);
  if (!layer->out) {
    perror("malloc");
    free(layer->z);
    return 1;
  }

  layer->biases = getTensor(2, shapeOut);
  if (!layer->biases) {
    perror("malloc");
    free(layer->out);
    free(layer->z);
    return 1;
  }

  layer->dBiases = getTensor(2, shapeOut);
  if (!layer->dBiases) {
    perror("malloc");
    free(layer->biases);
    free(layer->out);
    free(layer->z);
    return 1;
  }

  uint32_t shapeWeights[2] = {sizeOut, sizeIn};
  layer->weights = getTensor(2, shapeWeights);
  if (!layer->weights) {
    perror("malloc");
    free(layer->dBiases);
    free(layer->biases);
    free(layer->out);
    free(layer->z);
    return 1;
  }

  layer->dWeights = getTensor(2, shapeWeights);
  if (!layer->dWeights) {
    perror("malloc");
    free(layer->weights);
    free(layer->dBiases);
    free(layer->biases);
    free(layer->out);
    free(layer->z);
    return 1;
  }

  fillGaussTensor(layer->weights);
  fillGaussTensor(layer->biases);

  layer->in = NULL;
  layer->sizeIn = sizeIn;
  layer->sizeOut = sizeOut;
  layer->act = act;

  return 0;
}

layer_t *getLayer(uint32_t sizeIn, uint32_t sizeOut, act_t act) {
  layer_t *layer = (layer_t *)malloc(sizeof(layer_t));
  if (!layer) {
    perror("malloc");
    return NULL;
  }

  if (initLayer(layer, sizeIn, sizeOut, act)) {
    free(layer);
    return NULL;
  }

  return layer;
}

network_t *getNetwork(uint32_t sizeIn, uint32_t numLayers, uint32_t *sizeLayers) {
  network_t *network = (network_t *)malloc(sizeof(network_t));
  if (!network) {
    perror("malloc");
    return NULL;
  }

  network->sizeLayers = (uint32_t *)malloc(sizeof(uint32_t) * numLayers);
  if (!network->sizeLayers) {
    perror("malloc");
    free(network);
    return NULL;
  }

  network->layers = (layer_t *)malloc(sizeof(layer_t) * numLayers);
  if (!network->layers) {
    perror("malloc");
    free(network->sizeLayers);
    free(network);
    return NULL;
  }

  for (uint32_t i = 0; i < numLayers; i++) {
    network->sizeLayers[i] = sizeLayers[i];

    if (numLayers == 1) {
      initLayer(network->layers, sizeIn, sizeLayers[i], SOFT);
    } else if (i == 0) {
      initLayer(network->layers, sizeIn, sizeLayers[i], RELU);
    } else if (i == numLayers - 1) {
      initLayer(network->layers + i, sizeLayers[i - 1], sizeLayers[i], SOFT);
    } else {
      initLayer(network->layers + i, sizeLayers[i - 1], sizeLayers[i], RELU);
    }

    if (i > 0) {
      insertInput(network->layers + i, network->layers[i - 1].out);
    }
  }

  network->sizeIn = sizeIn;
  network->numLayers = numLayers;

  return network;
}

void freeLayerAtt(layer_t *layer) {
  if (!layer) {
    return;
  }
  freeTensor(layer->in);
  freeTensor(layer->z);
  freeTensor(layer->out);
  freeTensor(layer->weights);
  freeTensor(layer->biases);
  freeTensor(layer->dWeights);
  freeTensor(layer->dBiases);
}

void freeLayer(layer_t *layer) {
  if (!layer) {
    return;
  }
  freeLayerAtt(layer);
  free(layer);
}

void freeNetwork(network_t *network) {
  if (!network) {
    return;
  }
  for (uint32_t i = 0; i < network->numLayers; i++) {
    freeLayerAtt(network->layers + 1);
  }
  free(network->layers);
  free(network->sizeLayers);
  free(network);
}

int insertInput(layer_t *layer, tensor_t *input) {
  if (layer->sizeIn != input->len) {
    fprintf(stderr, "insertInput: dimention missmatch\n");
    return 1;
  }

  uint32_t shapeIn[] = {input->len, 1};
  layer->in = allocTensor(input->dim, shapeIn);
  if (!layer->in) {
    return 1;
  }

  input->storage->refCount++;
  layer->in->storage = input->storage;
  layer->in->data = input->data;

  return 0;
}

int computeLayer(layer_t *layer) {
  if (!layer) {
    return 1;
  }

  multiply2dTensor(layer->z, layer->weights, layer->in);
  addTensor(layer->z, layer->biases);
  switch (layer->act) {
    case RELU:
      reluTensor(layer->out, layer->z);
      break;
    case SOFT:
      softmaxTensor(layer->out, layer->z);
      break;
  }

  return 0;
}

int computeNetwork(network_t *network, tensor_t *input) {
  insertInput(network->layers, input);
  for (uint32_t i = 0; i < network->numLayers; i++) {
    computeLayer(network->layers + i);
  }

  return 0;
}

