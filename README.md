# Query-3DGS-LLM

This repository contains code for LLM-powered semantic querying of 3D Gaussian Splatting scenes. Built upon the [LEGaussian](link-to-original-repo) framework, this project extends the capabilities by integrating Large Language Models (LLMs) like Qwen and Llama for semantic scene understanding and manipulation.

## Overview

Our project enables semantic querying and manipulation of 3D Gaussian Splatting scenes using natural language through LLM integration. Users can query scene elements, modify representations, and generate targeted visualizations through natural language interactions.

## Features

- Integration with multiple LLM models (Qwen 2.5 and Llama series)
- Semantic scene understanding and querying
- Natural language based scene manipulation
- Support for multiple 3D scene datasets (mipnerf360, wayvescene)
- Evaluation metrics and visualization tools

## Installation

```bash
# Clone the repository
git clone https://github.com/AmirhoseinCh/Query-3DGS-LLM.git
cd Query-3DGS-LLM

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── arguments/          # Command line argument definitions
├── base/              # Base classes and core functionality
├── configs/           # Configuration files for different datasets
├── gaussian_renderer/ # Gaussian splatting renderer
├── preprocess/        # Data preprocessing utilities
├── scene/            # Scene representation and management
├── utils/            # Utility functions and helpers
└── requirements.txt   # Project dependencies
```

## Usage

### Training

```bash
# Run training script
./run.sh
```

### Evaluation

```bash
# Run evaluation
python eval.py --config configs/your_config.yaml
```

### Scene Rendering

```bash
# Render scenes
./render_scenes.sh
```

## Models

We support multiple LLM models:
- Qwen 2.5 series (0.5B, 1.5B, 3B, 7B)
- Llama series

## Results

Our evaluation metrics include:
- Mean accuracy comparison
- Mean IoU comparison
- Mean precision comparison

Detailed results and visualizations can be found in your paper/documentation.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{your-paper,
    title={},
    author={},
    journal={},
    year={}
}
```

## Acknowledgments

This work builds upon the [LEGaussian](link-to-original-repo) implementation. We thank the original authors for making their code available.

## License

[Your chosen license] - See LICENSE file for details

## Contact

[Your name] - [Your email/contact information]