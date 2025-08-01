# FedHSD: Hierarchical Self-Distillation for Personalized Federated Learning

This repository contains the official code for the paper **FedHSD: Hierarchical Self-Distillation for Personalized Federated Learning**.

We would like to express our gratitude to the **[PFLlib](https://github.com/TsingZ0/PFLlib)** library, upon which our method is based.

## How to Run

### 1. Environment Setup

Configure your environment using Conda:

```bash
conda env create -f env_cuda_latest.yaml
```

### 2. Data Partitioning

Navigate to the dataset directory and execute the following script to partition the Cifar100 dataset with a non-IID Dirichlet distribution:

```bash
cd ./dataset
python generate_Cifar100.py noniid - dir
```

### 3. Execution

Change to the system directory and run the main experiment script:

```bash
cd ./system
python main.py -data Cifar100 -ncl 100 -m CNN -algo FedHSD -gr 200 -did 0 -jr 1.0 -ls 5 -lbs 128 -bt 1.0 -lam 1.0 -sg 0.9
```

