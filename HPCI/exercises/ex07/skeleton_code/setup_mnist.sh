#!/bin/bash -l

FILES=()
# test set labels and images
FILES+=("t10k-labels-idx1-ubyte")
FILES+=("t10k-images-idx3-ubyte")
# training set labels and images
FILES+=("train-labels-idx1-ubyte")
FILES+=("train-images-idx3-ubyte")

for F in "${FILES[@]}"
do

if [ ! -f $F ]
then
  # download from source:
  wget yann.lecun.com/exdb/mnist/${F}.gz
  # unzip
  gunzip ${F}.gz
fi

done


