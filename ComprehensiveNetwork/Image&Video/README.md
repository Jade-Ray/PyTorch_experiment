# DL Samples for Image & Video Task

Various function about image and video DL with these samples:

- Detection

  - `ObjectDetection.py` : Detecte object in image with `FastRcnn`.
  
  - `DETR.py` : *End to End Object Detection with Transformer*, instead common detector address set prediction with proposals, anchors and postprocess, this method directly force unique preditions via bi-partite matching in non-local computations of Transformer.

- Transfer Learning

  - `transferLearning.py` : Using `TransferLearning` to fast train custom data.

- Adversarial Example Generation

  - `FastGradientSignAttack.py` : Using `FGSM` generate misleading image with little perturbation for human.

- Generative Adversarial Networks

  - `DCGAN.py` : Using `Encoder-Decoder` adversarial generate fake image likes origin image.

- Others

  - `NeuralStyleTransfer.py` : Training the specific image style merge to another input image.

  - `SpatialTransformerNetwork.py` : Learning spatial transform in image with `STN`.
