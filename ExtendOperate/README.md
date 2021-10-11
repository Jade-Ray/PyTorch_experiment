# Some Extended Operation of Pytorch

Exclusive and helpful operation with these samples:

- Datasets & Dataloaders

  - `CustomData.py` : Creating a Custom Dataset for your files

  - `DataParallelism.py` : Single machine with mutiple GPUs parallelism train

- Visualizing with TensorBoard

  - `Visualizing.py` : Visualizing Models, Data and Training in TensorBoard

- Save and Load Model

  - `saver_net_parma.py` : Saving and Loading Model Weights or Model with Shapes

- Model Optimization
  
  - `ProfilerAPIusage.py` : Introduce Pytorch Profiler API operation and improve memory and time with a custom net module in example
  
  - `ProfilerWithTensorboard.py` : Add profiler information with ResNet18 on FashionMNIST dataset to TensorBoard

- Parallel and Distributed Training

  - `Single_machine_model_parallel.py` : Introduce Model Parallel while the model too big to train, and speed up by pipeline inputs in Model Parallel

  - `Distributed_data_parallel.py`: Introduce how to using DistributedDataParallel(DDP) implements data parallelism at the module level which can run across multiple machines

  - `Distributed_application.py`: Introduce distributed package of PyTorch and how to set up the distributed setting with different communication strategies.
