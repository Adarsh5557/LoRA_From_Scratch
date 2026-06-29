# LoRA from Scratch in PyTorch

This project implements **LoRA (Low-Rank Adaptation)** from scratch in PyTorch and demonstrates how parameter-efficient fine-tuning can be applied to a neural network trained on the MNIST handwritten digit dataset.

LoRA is a popular Parameter-Efficient Fine-Tuning (PEFT) technique that freezes the original model weights and introduces small trainable low-rank matrices to adapt the model for new tasks. This significantly reduces the number of trainable parameters while retaining most of the model's original knowledge.

## Features

- Implementation of LoRA from scratch using PyTorch
- Custom LoRA parametrization module
- Integration with `torch.nn.utils.parametrize`
- Selective freezing of original model parameters
- Fine-tuning using only LoRA parameters
- Parameter count comparison between the original model and the LoRA-augmented model
- Verification that original weights remain unchanged during fine-tuning
- Training and evaluation on the MNIST dataset

## Project Workflow

1. Train a neural network on the MNIST dataset.
2. Save the original model weights.
3. Attach LoRA adapters to selected linear layers.
4. Freeze the original network parameters.
5. Train only the LoRA parameters.
6. Compare parameter counts and memory efficiency.
7. Verify that the original weights remain unchanged.
8. Evaluate the adapted model.

## Technologies Used

- Python
- PyTorch
- Torchvision
- tqdm

## Key Concepts Covered

- Low-Rank Adaptation (LoRA)
- Parameter-Efficient Fine-Tuning (PEFT)
- Neural Networks
- Transfer Learning
- Model Parametrization
- Fine-Tuning Large Models
- MNIST Classification

## Why LoRA?

Traditional fine-tuning updates every parameter in a neural network, which can be computationally expensive. LoRA instead learns a low-rank update to the weight matrices, allowing effective adaptation with only a small fraction of the original parameters.

## Results

This project demonstrates that a model can be adapted using a very small number of additional trainable parameters while keeping the original network weights frozen, showcasing the effectiveness of parameter-efficient fine-tuning techniques.
