# DL Samples for Image & Video Task

Various function about image and video DL with these samples:

- Classification

  - [SwinTransformer](SwinTransformer.py) : *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, proposed a hierarchical Transformer whose representation is computed with `Shifted Windows`. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size.

- Detection

  - [FastRcnn](ObjectDetection.py) : Detecte object in image with `FastRcnn`.
  
  - [DETR](DETR.py) : *End to End Object Detection with Transformer*, instead common detector address set prediction with proposals, anchors and postprocess, this method directly force unique preditions via bi-partite matching in non-local computations of Transformer.

- Transfer Learning

  - [TransferLearning](transferLearning.py) : Using `TransferLearning` to fast train custom data.

- Adversarial Example Generio

  - [FastGradientSignAttack](FastGradientSignAttack.py) : Using `FGSM` generate misleading image with little perturbation for human.

- Generative Adversarial Networks

  - [DCGAN](DCGAN.py) : Using `Encoder-Decoder` adversarial generate fake image likes origin image.

- Video Recognition

  - [SlowFast](SlowFast.py) : *SlowFast Nestwoks for Video Recognition*, involved a `Slow pathway` to capture spatial semantics and a `Fast pathway` to capture motion at fine temporal resolution.

- Others

  - [NeuralStyleTransfer](NeuralStyleTransfer.py) : Training the specific image style merge to another input image.

  - [SpatialTransformerNetwork](SpatialTransformerNetwork.py) : Learning spatial transform in image with `STN`.
