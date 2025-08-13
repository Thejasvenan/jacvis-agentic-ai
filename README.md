# Auto-Adaptive Fine-tuning for Jac MTLLM using RPG Game Generation

## Overview
This project aims to create an intelligent **MTLLM** (Multi-Task Large Language Model) system that **automatically collects training data** from large model interactions and **continuously fine-tunes smaller, local models** to replace them.  
The use case focuses on **RPG game level generation** — a complex structured data generation task that requires spatial awareness, adherence to game rules, and creativity.

By progressively training local models (like TinyLLaMA) using real interaction data, the system reduces **cost**, **latency**, and **dependency** on external APIs.

---

## Problem Statement
Current MTLLM setups for structured data generation suffer from:

1. **High API Costs** – Continuous use of large cloud models is expensive.  
2. **Latency Issues** – Network calls cause delays in real-time applications.  
3. **Dependency Risk** – Reliance on third-party services and pricing policies.

---

## Proposed Solution

### 1. Dataset Generation Using Large Models
- **Complex Use Case**: RPG game level generation requiring:
  - Understanding of game mechanics and spatial layout
  - Creating interconnected game objects (players, enemies, terrain)
  - Following strict playability and structural rules
- **Automated Pipeline**:
  - Use Jac abilities with `by <llm>` calls to GPT-4, Claude, Gemini Pro
  - Generate multiple versions to build a **diverse dataset**
  - Store original prompts and structured outputs
- **Optional Filtering**:
  - Validate levels for:
    - Player spawn points
    - Proper enemy placement
    - Reachability of all areas
    - Structural integrity

---

### 2. Small LLM Training Pipeline
- **Target Model**: TinyLLaMA for local deployment  
- **Fine-tuning Techniques**:
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized LoRA) for efficiency
- **Optimization**:
  - Apply quantization for smaller model size and faster inference
  - Use dataset generated from `by <llm>` calls

---

### 3. MTLLM Plugin Integration
- **Automatic Data Collection**:
  - Capture prompt-response pairs from successful large model calls
- **Dynamic Training**:
  - Periodic background fine-tuning during low usage periods
- **Intelligent Model Switching**:
  - Replace large model calls with local models when confidence is high
  - Maintain per-call-site specialized models
- **Persistence**:
  - Store, cache, and version trained models
  - Rollback capability

---

### 4. Evaluation Framework
- **Manual Correctness Checks**:
  - Create evaluation rubrics for:
    - Gameplay fun & viability
    - Structural correctness
    - Adherence to constraints & themes
- **Comparison**:
  - Compare large model vs. fine-tuned local model outputs

---

## Benefits
- **Cost Reduction** – Use large models only for initial dataset generation  
- **Lower Latency** – Run fine-tuned models locally for real-time use  
- **Resilience** – Less dependent on third-party API pricing & uptime  
- **Continuous Improvement** – Automatic self-learning from live interactions  

---

## Tech Stack
- **Language**: [Jac Programming Language](https://www.jac-lang.org/)
- **Models**: GPT-4, Claude, Gemini Pro, TinyLLaMA
- **Fine-tuning**: LoRA, QLoRA
- **Deployment**: Jac MTLLM Plugin, Local Model Hosting
- **Evaluation**: Custom rule-based & manual rubric system
