# KAN Emotional Character with LLaMA 3.1 8B Instruct

## Overview

This project implements an Emotional AI Character using the LLaMA 3.1 8B Instruct model, enhanced with a Knowledge Augmentation Network (KAN) and an emotional state system. The character can engage in conversations, remember past interactions, and dynamically adjust its emotional state based on user feedback.

## Features

- Utilizes the LLaMA 3.1 8B Instruct model for natural language processing
- Implements a Knowledge Augmentation Network (KAN) for enhanced learning
- Includes an emotional state system that evolves based on interactions
- Maintains a memory of past conversations for contextual responses
- Provides a graphical user interface for easy interaction

## Requirements

- Python 3.8 or later
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- Internet connection for initial setup and model download

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/waefrebeorn/KAN-WuBu-Memory.git
   cd kan-WuBu-Memory
   ```

2. Run the `setup_and_run.bat` script:
   ```
   setup_and_run.bat
   ```
   This script will set up a virtual environment, install dependencies, and guide you through the model download process.

## Manual Model Download

Due to licensing restrictions, you need to manually download the LLaMA 3.1 8B Instruct model files:

1. Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Accept the license agreement if you haven't already.
3. Download the following files:
   - config.json
   - model-00001-of-00004.safetensors
   - model-00002-of-00004.safetensors
   - model-00003-of-00004.safetensors
   - model-00004-of-00004.safetensors
   - tokenizer.json
   - tokenizer_config.json
4. Place these files in the `models/Meta-Llama-3.1-8B-Instruct` directory.

## Usage

After completing the setup and model download:

1. Run the application:
   ```
   python kan_gui.py
   ```

2. In the GUI:
   - Enter a character description when prompted
   - Interact with the character by typing messages
   - Provide emotional feedback after each interaction

3. To end the session, type 'exit' or close the GUI window.

## Project Structure

- `kan_emotional_character_llama_hf.py`: Main implementation of the KAN Emotional Character
- `kan_gui.py`: Graphical user interface for interacting with the character
- `setup_and_run.bat`: Setup script for Windows
- `requirements.txt`: List of Python dependencies
- `models/`: Directory to store the downloaded model files

## Customization

- Modify the `EmotionalState` class in `kan_emotional_character_llama_hf.py` to adjust emotional dynamics
- Tweak the `KAN` class to experiment with different network architectures
- Adjust generation parameters in the `generate_response` method for different text outputs

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try reducing the model's precision or using a smaller model
- For other issues, check the console output for error messages and refer to the Hugging Face Transformers documentation

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The LLaMA 3.1 8B Instruct model is provided by Meta AI
- This project uses the Hugging Face Transformers library

## Disclaimer

This project is for educational and research purposes only. Ensure you comply with all licensing terms when using the LLaMA model.
