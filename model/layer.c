#include "layer.h"
#include "tensor.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

layer_t *getLayer(uint32_t sizeIn, uint32_t sizeOut) {
  layer_t *layer = (layer_t *)malloc(sizeof(layer_t));
  if (!layer) {
    perror("malloc");
    return NULL;
  }

  uint32_t shapeOut[2] = {sizeOut, 1};
  layer->z = getTensor(2, shapeOut);
  if (!layer->z) {
    perror("malloc");
    free(layer);
    return NULL;
  }

  layer->out = getTensor(2, shapeOut);
  if (!layer->out) {
    perror("malloc");
    free(layer->z);
    free(layer);
    return NULL;
  }

  layer->biases = getTensor(2, shapeOut);
  if (!layer->biases) {
    perror("malloc");
    free(layer->out);
    free(layer->z);
    free(layer);
    return NULL;
  }

  layer->dBiases = getTensor(2, shapeOut);
  if (!layer->dBiases) {
    perror("malloc");
    free(layer->biases);
    free(layer->out);
    free(layer->z);
    free(layer);
    return NULL;
  }

  uint32_t shapeWeights[2] = {sizeOut, sizeIn};
  layer->weights = getTensor(2, shapeWeights);
  if (!layer->weights) {
    perror("malloc");
    free(layer->dBiases);
    free(layer->biases);
    free(layer->out);
    free(layer->z);
    free(layer);
    return NULL;
  }

  layer->dWeights = getTensor(2, shapeWeights);
  if (!layer->dWeights) {
    perror("malloc");
    free(layer->weights);
    free(layer->dBiases);
    free(layer->biases);
    free(layer->out);
    free(layer->z);
    free(layer);
    return NULL;
  }

  layer->in = NULL;
  layer->sizeIn = sizeIn;
  layer->sizeOut = sizeOut;

  return layer;
}

void freeLayer(layer_t *layer) {
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

  return 0;
}
