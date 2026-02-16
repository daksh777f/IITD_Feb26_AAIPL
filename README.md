# AMD AI Premier League (AAIPL) - Deterministic Competitive Answering Engine

## What We Built

This project implements a competitive AI system for the AMD AI Premier League tournament, consisting of two specialized agents:

1. **Question Agent (Q-Agent)**: An AI system that generates challenging puzzle-based questions across various topics including mathematics, logic, physics, chemistry, computer science, and general reasoning.

2. **Answer Agent (A-Agent)**: An AI system that analyzes and answers multiple-choice questions with high accuracy, providing reasoning for its answers.

Both agents compete in a tournament format where teams face off in 1v1 matches, with each team's Q-Agent trying to generate difficult questions while their A-Agent attempts to answer the opponent's questions correctly.

## Techniques We Used

### Model Architecture
- **Base Model**: Qwen3-4B, a 4-billion parameter pre-trained language model
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) for efficient model customization
- **Framework**: Unsloth for accelerated training on AMD hardware
- **Inference Optimization**: Batch processing for efficient text generation

### Training Approach
- **Synthetic Data Generation**: Created custom training datasets for both agents
- **Prompt Engineering**: Carefully crafted system prompts to guide model behavior
- **Parameter-Efficient Fine-Tuning**: Used LoRA adapters to fine-tune the model without modifying the base weights
- **Deterministic Behavior**: Implemented seed control for reproducible results

### Key Technologies
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for model loading and inference
- **PEFT**: Parameter-Efficient Fine-Tuning library for LoRA implementation
- **AMD ROCm**: Hardware acceleration for training and inference on AMD GPUs

## How We Built It

### 1. Environment Setup
- Configured PyTorch with AMD ROCm support for GPU acceleration
- Set up Hugging Face model caching in a local workspace directory
- Installed Unsloth for efficient LoRA fine-tuning

### 2. Question Agent Development
**Implementation Steps:**
- Loaded the Qwen3-4B base model with appropriate caching configuration
- Implemented batch processing for efficient question generation
- Created custom prompts that guide the model to generate:
  - Questions relevant to specified topics
  - Multiple-choice options (A, B, C, D)
  - Correct answers with explanations
  - Proper JSON formatting

**Technical Details:**
- Used chat template formatting with system and user prompts
- Implemented token generation with configurable parameters (temperature, top_p, repetition penalty)
- Added timing mechanisms to ensure questions are generated within the 13-second limit
- Applied padding and truncation for batch processing

### 3. Answer Agent Development
**Implementation Steps:**
- Built on the same Qwen3-4B foundation as the Q-Agent
- Integrated LoRA adapter loading for fine-tuned weights
- Implemented answer extraction and reasoning generation
- Created fallback mechanisms for when LoRA adapters are unavailable

**Technical Details:**
- Configured left-padding for proper batch inference
- Used PEFT library to merge LoRA adapters with the base model
- Implemented deterministic seed setting for consistent results
- Created prompts that encourage step-by-step reasoning
- Added JSON parsing and validation for answer extraction

### 4. Fine-Tuning Process
**Synthetic Data Generation:**
- Created training examples in conversational format
- Generated diverse question-answer pairs across all required topics
- Formatted data according to model chat templates

**LoRA Training Configuration:**
- Target modules: q_proj, k_proj, v_proj, o_proj (attention layers)
- LoRA rank: 16 (balance between performance and efficiency)
- LoRA alpha: 32 (scaling factor)
- Batch size: 32 (leveraging AMD MI300X's 192GB HBM3 memory)
- Gradient accumulation: Used for effective larger batch sizes
- Learning rate: Optimized for LoRA fine-tuning

### 5. Optimization and Validation
- Implemented format validation to ensure outputs match required JSON schemas
- Added error handling and fallback mechanisms
- Tuned generation parameters (max_tokens, temperature, etc.)
- Tested against format requirements and sample questions
- Validated model checkpoints save and load correctly

### 6. Deployment Structure
**Project Organization:**
```
IITD_Feb26_AAIPL/
├── agents/
│   ├── question_model.py       # Q-Agent implementation
│   ├── question_agent.py       # Q-Agent wrapper
│   ├── answer_model.py         # A-Agent implementation
│   └── answer_agent.py         # A-Agent wrapper
├── assets/
│   ├── topics.json             # Competition topics
│   ├── sample_question.json    # Expected question format
│   └── sample_answer.json      # Expected answer format
├── utils/
│   └── build_prompt.py         # Prompt engineering utilities
├── hf_models/                  # Model cache directory
├── agen.yaml                   # A-Agent configuration
├── qgen.yaml                   # Q-Agent configuration
└── tutorial.ipynb              # Training tutorial
```

## Model Performance Characteristics

### Question Agent
- Generates questions within 13 seconds
- Produces valid JSON formatted output
- Creates diverse questions across multiple topics
- Balances difficulty to maximize scoring potential

### Answer Agent
- Processes answers within 9 seconds
- High accuracy on puzzle-based questions
- Provides reasoning for each answer
- Handles edge cases and malformed inputs gracefully

## Competition Strategy

The system is designed for a tournament where:
- Each match consists of two innings (like cricket)
- Q-Agent score is based on questions the opponent's A-Agent fails to answer
- A-Agent score is based on correctly answering opponent's questions
- Total team score combines Q-Agent and A-Agent performance

Our implementation focuses on:
1. Generating challenging but fair questions
2. Maximizing answer accuracy through fine-tuning
3. Ensuring format compliance for all outputs
4. Optimizing inference speed for real-time competition

## Technical Constraints

- AMD ROCm-compatible hardware required
- Models must be from the provided AMD AI Academy cache
- No RAG (Retrieval Augmented Generation) allowed
- No adversarial techniques permitted
- English language only
- Strict token limits enforced via YAML configuration files

## Future Improvements

Potential enhancements for this system include:
- Advanced prompt engineering techniques
- Multi-task learning for both agents
- Ensemble approaches for answer validation
- Dynamic difficulty adjustment for question generation
- Advanced reasoning techniques (chain-of-thought, tree-of-thought)
