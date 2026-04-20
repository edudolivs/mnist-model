#include "layer.h"
#include "tensor.h"
#include "loader.h"
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

  fillTensor(layer->dBiases, 0);
  fillTensor(layer->dWeights, 0);
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

network_t *getNetwork(uint32_t sizeIn, uint32_t numLayers, uint32_t *sizeLayers, float learningRate, uint32_t batchSize) {
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

  uint32_t maxSize = 0;
  for (uint32_t i = 0; i < numLayers; i++) {
    network->sizeLayers[i] = sizeLayers[i];
    if (sizeLayers[i] > maxSize) {
      maxSize = sizeLayers[i];
    }

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

  network->buffers = (float **)malloc(sizeof(float *) * 2);
  network->buffers[0] = (float *)malloc(sizeof(float) * maxSize);
  network->buffers[1] = (float *)malloc(sizeof(float) * maxSize);

  network->sizeIn = sizeIn;
  network->numLayers = numLayers;
  network->learningRate = learningRate;
  network->batchSize = batchSize;

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
    freeLayerAtt(network->layers + i);
  }
  free(network->layers);
  free(network->sizeLayers);
  free(network);
}

int zeroNetworkDerivative(network_t *network) {
  for (uint32_t i = 0; i < network->numLayers; i++) {
    layer_t *layer = network->layers + i;
    fillTensor(layer->dWeights, 0);
    fillTensor(layer->dBiases, 0);
  }

  return 0;
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

int accDeriveZ(layer_t *layer, float *dOut) {
  switch (layer->act) {
    case SOFT: {
      break;
    }
    case RELU: {
      for (uint32_t i = 0; i < layer->sizeOut; i++) {
        if (layer->z->data[i] < 0) {
          dOut[i] = 0;
        } else if (layer->z->data[i] == 0) {
          dOut[i] *= 0.5;
        }
      }
      break;
    }
  }

  return 0;
}

int accDeriveBiases(layer_t *layer, float *dOut) {
  for (uint32_t i = 0; i < layer->sizeOut; i++) {
    layer->dBiases->data[i] += dOut[i];
  }

  return 0;
}

int accDeriveWeights(layer_t *layer, float *dOut) {
  for (uint32_t i = 0; i < layer->sizeIn; i++) {
    for (uint32_t j = 0; j < layer->sizeOut; j++) {
      *getValue(layer->dWeights, j, i) += dOut[j] * layer->in->data[i];
    }
  }

  return 0;
}

int accDeriveWO(layer_t *layer, float *dIn, float *dOut) {
  for (uint32_t i = 0; i < layer->sizeIn; i++) {
    dIn[i] = 0;
    for (uint32_t j = 0; j < layer->sizeOut; j++) {
      *getValue(layer->dWeights, j, i) += dOut[j] * layer->in->data[i];
      dIn[i] += *getValue(layer->weights, j, i) * dOut[j];
    }
  }

  return 0;
}

int accDeriveNetwork(network_t *network, uint8_t label) {

  layer_t *lastLayer = (network->layers + network->numLayers - 1);

  switch(lastLayer->act) {
    case SOFT: {
      for (uint32_t i = 0; i < lastLayer->sizeOut; i++) {
        network->buffers[(network->numLayers - 1) % 2][i] = lastLayer->out->data[i] - (i == label);
      }
      break;
    }
    case RELU: {
      break;
    }
  }

  for (uint32_t i = network->numLayers; i-- > 0; ) {
    accDeriveZ(network->layers + i, network->buffers[i % 2]);
    accDeriveBiases(network->layers + i, network->buffers[i % 2]);
    if (i == 0) {
      accDeriveWeights(network->layers + i, network->buffers[i % 2]);
    } else {
      accDeriveWO(network->layers + i, network->buffers[(i + 1) % 2], network->buffers[i % 2]);
    }
  }

  return 0;
}

int updateNetwork(network_t *network) {
  for (uint32_t i = 0; i < network->numLayers; i++) {
    layer_t *layer = network->layers + i;
    for (uint32_t j = 0; j < layer->weights->len; j++) {
      layer->weights->data[j] -= layer->dWeights->data[j];
    }
    for (uint32_t j = 0; j < layer->biases->len; j++) {
      layer->biases->data[j] -= (network->learningRate * layer->dBiases->data[j]) / network->batchSize;
    }
  }
  zeroNetworkDerivative(network);

  return 0;
}

int trainEpoch(network_t *network, shuffler_t *shuffler) {
  shuffleData(shuffler);
  
  for (uint32_t i = 0; i < shuffler->lImages->labels->len; i++) {
    computeNetwork(network, shuffler->views + i);
    accDeriveNetwork(network, shuffler->sLabels[i]);
    if (i % network->batchSize == network->batchSize - 1) {
      updateNetwork(network);
    }
  }

  return 0;
}
