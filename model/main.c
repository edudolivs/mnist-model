#include "loader.h"
#include "layer.h"
#include "tensor.h"
#include "random.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

int loadMnist() {
  // char *trainImagesPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-images.idx3-ubyte";
  // char *trainLabelsPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-labels.idx1-ubyte";
  //
  // tensor_t *trainImages = loadIdx(trainImagesPath);
  // tensor_t *trainLabels = loadIdx(trainLabelsPath);
  //
  // uint32_t imageId = 0;
  //
  // uint32_t *indexes = getIndexesArray(trainImages);
  // tensor_t *views = getViewArray(trainImages);
  // float *labels = (float *)malloc(sizeof(float) * trainLabels->len);
  //
  // shuffleData(views, labels, trainImages, trainLabels, indexes);
  //
  // printf("label: %.0f\n", labels[imageId]);
  // displayImage(views + imageId);
  //
  // tensor_t *view = getView(trainImages, indexes[0]);
  //
  // printf("label %u: %.0f\n", indexes[0], trainLabels->data[indexes[0]]);
  // displayImage(view);
  //
  // printf("%u\n", indexes[500]);
  //
  // freeTensor(trainImages);
  // freeTensor(trainLabels);

   return 0;
}

int testOperations() {
  uint32_t shape[] = {3, 3};
  tensor_t *a = getTensor(2, shape);
  fillGaussTensor(a);
  display2dTensor(a);
  tensor_t *b = getTensor(2, shape);
  fillGaussTensor(b);
  display2dTensor(b);

  tensor_t *prod = getTensor(a->dim, a->shape);
  multiply2dTensor(prod, a, b);
  display2dTensor(prod);

  float filler = (float)(randUint32() % 3) - 1;
  fillTensor(a, filler);
  display2dTensor(a);
  addTensor(prod, a);

  display2dTensor(prod);

  freeTensor(b);
  freeTensor(a);
  freeTensor(prod);

  return 0;
}

int testLayers() {
  uint32_t shapeIn[] = {2, 1};
  tensor_t *input = getTensor(2, shapeIn);
  fillGaussTensor(input);
  printf("input:\n");
  display2dTensor(input);

  uint32_t sizeLayers[] = {10, 10};
  network_t *network = getNetwork(shapeIn[0], 2, sizeLayers, 1, 100);

  computeNetwork(network, input);

  for (uint32_t i = 0; i < network->numLayers; i++) {
    printf("layer %u weights:\n", i + 1);
    display2dTensor(network->layers[i].weights);
    printf("layer %u biases:\n", i + 1);
    display2dTensor(network->layers[i].biases);
    printf("layer %u output:\n", i + 1);
    display2dTensor(network->layers[i].out);
  }

  freeNetwork(network);
  freeTensor(input);

  return 0;
}

int testTrain() {
  char *trainImagesPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-images.idx3-ubyte";
  char *trainLabelsPath = "/Users/edudolivs/Projects/mnist-model/mnist-dataset/train-labels.idx1-ubyte";

  lImages_t *lImages = getLabeledImages(trainImagesPath, trainLabelsPath);
  shuffler_t *shuffler = getShuffler(lImages);

  uint32_t sizeLayers[] = {10, 10};
  network_t *network = getNetwork(lImages->images->stride[0], 2, sizeLayers, 0, 100);

  float correct = 0;
  uint32_t max;
  for (uint32_t i = 0; i < 60000; i++) {
    computeNetwork(network, shuffler->views + i);
    max = 0;
    for (uint32_t j = 1; j < 10; j++) {
      if (network->layers[1].out->data[j] > network->layers[1].out->data[max]) {
        max = j;
      }
    }
    if (max == shuffler->sLabels[i]) {
      correct++;
    }
  }
  printf("accuracy before: %f\n", correct / 60000.0);

  for (uint32_t i = 0; i < 10; i++) {
    trainEpoch(network, shuffler);
    printf("finished epoch %d\n", i + 1);
    correct = 0;
    for (uint32_t i = 0; i < 60000; i++) {
      computeNetwork(network, shuffler->views + i);
      max = 0;
      for (uint32_t j = 1; j < 10; j++) {
        if (network->layers[1].out->data[j] > network->layers[1].out->data[max]) {
          max = j;
        }
      }
      if (max == shuffler->sLabels[i]) {
        correct++;
      }
    }
    printf("accuracy: %.3f%%\n", 100 * correct / 60000.0);

  }

  return 0;
}

int main() {
  seed(time(NULL));

  testTrain();

  return 0;
}
