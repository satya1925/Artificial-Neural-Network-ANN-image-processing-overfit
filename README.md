# Artificial Neural Network (ANN) for Character Recognition

This project implements a **supervised learning** approach using a **multilayer feed-backward artificial neural network (ANN)** to recognize alphabetic characters represented in **5x6 pixel grids**. The model achieves an exceptional accuracy of **~99.9999%**, demonstrating its effectiveness in character recognition tasks.

## Project Overview

- **Objective**: Predict alphabetic characters from 5x6 binary pixel inputs.
- **Model**: Custom-built ANN with backpropagation for training.
- **Accuracy**: Achieves near-perfect accuracy (~99.9999%) on the training dataset.

## Repository Contents

- `abcd.py`: Script for data preprocessing and preparation.
- `ann.py`: Implementation of the ANN model, including training and evaluation routines.
- `weights.json`: Serialized weights of the trained ANN model.
- `alfabetsmachinlearnig.html`: Visualization of training metrics and model performance.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the following packages:

- `numpy`
- `matplotlib` (optional, for visualizations)

You can install the required packages using pip:

```bash
pip install numpy matplotlib
