<div align="center">

# HC-FMTL Visualization Analytics System

[![TypeScript](http://img.shields.io/badge/TypeScript-5.x-blue.svg)](https://www.typescriptlang.org/)
[![TensorFlow.js](http://img.shields.io/badge/TensorFlow.js-4.x-orange.svg)](https://www.tensorflow.org/js)
[![License](http://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

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
