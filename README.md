<div align="center">

# HC-FMTL Visualization Analytics System

[![TypeScript](http://img.shields.io/badge/TypeScript-5.x-blue.svg)](https://www.typescriptlang.org/)
[![TensorFlow.js](http://img.shields.io/badge/TensorFlow.js-4.x-orange.svg)](https://www.tensorflow.org/js)
[![License](http://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

<img width="1813" height="808" alt="image" src="https://github.com/user-attachments/assets/5b344da8-677d-461f-9dcd-a189820d3cd6" />

## Done By
**Name:** Yong Shao En Ernest  
**Matriculation Number:** U2221153B

---

## Description

This project presents a browser-based progressive visualization analytics system for controlled experimentation with heterogeneous-client federated multi-task learning (HC-FMTL). The system is implemented using TypeScript and TensorFlow.js, enabling interactive experimentation directly in the browser without requiring a separate backend training stack.

The platform is designed for teaching, prototyping, and qualitative comparison of federated learning behaviour under controlled settings.

---

## Core Features

- **Browser-based federated multi-task learning simulator**
- **TensorFlow.js implementation** of a lightweight dense-prediction model
- **Synthetic SHAPES dataset** for controlled non-IID experiments
- Support for multiple client heterogeneity settings:
  - Label skew
  - Quantity skew
  - Task type overlap
  - Task count imbalance
- Support for multiple aggregation methods:
  - **FedAvg**
  - **FedHCA²**
- **Round-level telemetry logging** for training and aggregation diagnostics
- **Three coordinated views** for analysis:
  - **Client View**
  - **Server View**
  - **Experiment View**
- Progressive visualization of:
  - training curves,
  - loss behaviour,
  - client update statistics,
  - aggregation behaviour,
  - qualitative task outputs

---

## Technology Stack

- **TypeScript**
- **TensorFlow.js**
- **HTML/CSS**
- **JavaScript tooling via npm**
- Browser-based visualization and interactive experiment controls

---

## Implemented System Components

### 1. Synthetic SHAPES Dataset
A configurable synthetic dataset used to simulate dense prediction tasks under controlled heterogeneity settings.

### 2. Lightweight Multi-Task Dense Prediction Model
A browser-friendly neural network architecture used for multi-task learning experiments.

### 3. Federated Aggregation Algorithms
- **FedAvg** for standard federated averaging
- **FedHCA²** for heterogeneity-aware conflict-averse aggregation

### 4. Telemetry and Diagnostics
The system records per-round metrics to support visualization and comparison of algorithmic behaviour.

### 5. Coordinated Visualization Views
- **Client View**: inspect per-client training behaviour and qualitative outputs
- **Server View**: inspect aggregation dynamics and client-level summaries
- **Experiment View**: compare completed runs across algorithms and heterogeneity scenarios

---

## How to Run

First, clone the repository and install dependencies:

```bash
git clone https://github.com/ernyo/HC-FMTL-Visualization-Analytics-System.git
cd HC-FMTL-Visualization-Analytics-System
npm install

# Start the development server:
npm run dev

# Build the project for production:
npm run build
```

## Project Structure

```text
HC-FMTL-Visualization-Analytics-System/
├── Experiments/                         # Experiment pipeline, scripts, logs, and ablation outputs
│   ├── ablation/
│   │   ├── architecture/               # Architecture ablation results by heterogeneity type/severity
│   │   │   ├── label_skew/
│   │   │   ├── quantity_skew/
│   │   │   ├── task_count_imbalance/
│   │   │   └── task_type_overlap/
│   │   ├── cac/                        # Conflict-averse strength ablation
│   │   ├── epochs_per_client/          # Local epoch sensitivity experiments
│   │   ├── number_of_samples/          # Sample-size sensitivity experiments
│   │   └── number_of_tasks/            # Task-count sensitivity experiments
│   ├── logs/                           # Experiment logs
│   ├── notebooks/                      # Analysis notebooks
│   ├── shell/                          # Shell scripts for running experiments
│   └── src/
│       ├── datasets/                   # Dataset utilities for experiments
│       ├── experiments/                # Experiment definitions and runners
│       ├── federated/                  # Federated learning logic
│       ├── models/                     # Experimental model implementations
│       └── utils/                      # Shared experiment utilities
├── Federated Learning Playground/      # Main browser-based application
│   ├── .github/                        # GitHub/project metadata
│   ├── dist/                           # Production build output
│   ├── images/                         # Project images and assets
│   ├── public/                         # Public static files
│   └── src/
│       ├── backend/                    # Application-side training / orchestration logic
│       ├── datasets/                   # Dataset generation and loading
│       ├── drivers/                    # Execution/control drivers
│       ├── federated/                  # Aggregation and federated training code
│       ├── models/                     # TensorFlow.js model definitions
│       ├── tests/                      # Test files
│       ├── utils/                      # Shared utilities
│       ├── views/                      # Client, Server, and Experiment views
│       └── visualizations/             # Charts and visualization components
├── POC/                                # Early prototypes and proof-of-concept implementations
│   ├── custom implementation/
│   │   └── SHAPES/                     # Synthetic SHAPES dataset assets
│   ├── dataset generation/             # Dataset generation experiments
│   └── hca2 algorithm/                 # HCA² prototype and reference implementation
│       ├── configs/
│       ├── datasets/
│       │   ├── NYUDv2/
│       │   ├── SHAPES/
│       │   └── utils/
│       ├── evaluation/
│       └── models/
└── Requirements Elicitation/           # Requirements and planning materials
```

---

## Ablation Studies

A major component of this project is comprehensive ablation studies to study algorithmic behaviour for each model architecture and hyperparameter setting. Systematic experiments were conducted to evaluate the impact of various hyperparameters on algorithmic performance.

### Running Ablation Experiments
```bash
sbatch run.sh

# Label skew
sbatch shell/label_skew_low.sh
sbatch shell/label_skew_med.sh
sbatch shell/label_skew_high.sh

# Quantity skew
sbatch shell/quantity_skew_low.sh
sbatch shell/quantity_skew_med.sh
sbatch shell/quantity_skew_high.sh
```

## Project Goals

This system was built to support the following goals:
- make federated multi-task learning behaviour more interpretable,
- provide a controlled environment for studying heterogeneity,
- support qualitative and quantitative comparison of aggregation methods,
- provide an educational tool for understanding federated learning dynamics in the browser.

The system is intended for:
- small-scale controlled experiments
- interactive analysis
- teaching and demonstration
- qualitative comparison of algorithmic behaviour

It is not intended as a production-scale federated training platform, but as an exploratory visualization and experimentation system.

---

## Acknowledgments

This project was developed as a Final Year Project focused on federated multi-task learning visualization and experimentation.
