#!/bin/bash

echo "REMOVING /tmp/datasets"
rm -rf /tmp/datasets
echo "MAKING /tmp/datasets/coco"
mkdir -p /tmp/datasets/coco
echo "ENTERING /tmp/datasets/coco"
pushd /tmp/datasets/coco

df

echo "COPYING DATA"
cp /scratch/05714/jgpaul/datasets/coco/coco_annotations_minival.tgz .
tar xzf coco_annotations_minival.tgz
rm coco_annotations_minival.tgz

cp /scratch/05714/jgpaul/datasets/coco/train2014.zip .
unzip -q train2014.zip
rm train2014.zip

cp /scratch/05714/jgpaul/datasets/coco/val2014.zip .
unzip -q val2014.zip
rm val2014.zip

cp /scratch/05714/jgpaul/datasets/coco/annotations_trainval2014.zip .
unzip -q annotations_trainval2014.zip
rm annotations_trainval2014.zip

popd

echo "Finished copying data to /tmp/datasets/coco"
