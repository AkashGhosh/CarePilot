# Agentic Pipeline

Agent-based system for medical visual tasks using target and critic agents with hierarchical reflection.

## Overview

The agentic pipeline orchestrates task processing through:
- **Target Agent**: Proposes actions based on current state, memory, and tools
- **Critic Agent**: Provides hierarchical reflection (action-level, trajectory-level, global)
- **Visual Tools**: Object detection, depth estimation, edge detection, zooming, visual grounding
- **Memory System**: Short-term and long-term memory for tracking progress
- **Task Controller**: Orchestrates the entire pipeline

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export HF_TOKEN="your_huggingface_token"
export DEEPINFRA_TOKEN="your_deepinfra_token"
```

## Configuration

Edit `config.py` to configure:
- Model names (target and critic agents)
- API endpoints
- Thresholds and decision parameters
- Dataset settings

## Usage

### Basic Usage

```python
from controllers.task_controller import TaskController

# Create task controller
controller = TaskController(use_mock=True)

# Process a single task
result = controller.process_single_task(
    user_goal="Load the MRI scan, create a segmentation of the tumor, and measure its volume."
)
```

### Command Line Usage

```bash
# Run with a single task
python main.py --mode original --goal "Load the MRI scan, create a segmentation of the tumor, and measure its volume."

# Run with dataset
python main.py --mode dataset --max_tasks 5

# Run with enhanced components
python main.py --mode enhanced --goal "Load the MRI scan, create a segmentation of the tumor, and measure its volume."
```

### Command Line Arguments

- `--mode`: Implementation mode (`original`, `enhanced`, `revpt`, `integration`, `dataset`)
- `--goal`: User goal/task to execute
- `--mock`: Use mock implementations for tools
- `--max_tasks`: Maximum number of tasks to process from dataset
- `--start_task`: Starting task index (0-indexed)
- `--dataset`: HuggingFace dataset name to use
- `--token`: HuggingFace API token
- `--verbose`: Enable verbose logging
- `--no-tools`: Disable tool calls

## Architecture

```
Agentic_Pipeline/
├── agents/              # Target and feedback agents
│   ├── target_agent.py  # Proposes actions
│   └── feedback_agent.py # Provides hierarchical reflection
├── controllers/         # Task control
│   └── task_controller.py # Pipeline orchestration
├── tools/               # Visual and medical tools
│   ├── tool_executor.py # Tool execution management
│   ├── visual_tools.py  # REVPT-style visual tools
│   └── medical_tools.py # Medical-specific tools
├── memory/              # Memory management
│   ├── short_term.py    # Short-term memory
│   ├── long_term.py     # Long-term memory
│   └── memory_manager.py # Memory coordination
├── evaluation/          # Action evaluation
│   ├── action_verifier.py # Programmatic verification
│   └── reflection.py    # Hierarchical reflection
├── utils/               # Utilities
│   ├── common.py        # Shared utilities
│   ├── logging_utils.py # Logging and monitoring
│   └── model_client.py  # Model API client
├── data/                # Data management
│   └── dataset_loader.py # Dataset loading
├── config.py            # Configuration
└── main.py              # Entry point
```

## Key Features

- **Visual Tool Integration**: Object detection, depth estimation, edge detection, zooming, visual grounding
- **Target-Critic Architecture**: Target agent proposes actions, critic agent provides feedback
- **Hierarchical Reflection**: Action-level, trajectory-level, and global reflection
- **Memory Management**: Short-term memory for recent actions, long-term memory for trajectory learning
- **Configurable Pipeline**: Easily configure thresholds, tools, and behaviors

## Dataset

The system uses HuggingFace datasets. Configure the dataset name in `config.py`:
- `rishuKumar404/test_cvpr_dataset` (default)
- `rishuKumar404/train_cvpr_dataset`
- `rishuKumar404/3dslicer-tabular-benchmark`

## Testing

Run tests with:
```bash
python -m unittest discover -s tests
```

