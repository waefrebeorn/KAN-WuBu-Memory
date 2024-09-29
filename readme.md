# KAN-WuBu-Memory: LLaMA 3.2 1B Instruct with Kolmogorov-Arnold Networks (KAN) Integration

![KAN-WuBu Memory](https://img.shields.io/badge/PyTorch-CUDA_Enabled-blue.svg)
![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen.svg)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange.svg)

## Project Overview

**KAN-WuBu-Memory** is an innovative project that combines the **LLaMA 3.2 1B** language model with **Kolmogorov-Arnold Networks (KAN)** to develop an emotionally aware, conversational memory system. This system builds on the idea of dynamic conversation, enabling **LLaMA 3.2 1B** to remember, adapt, and learn from interactions over time. It's designed to function as a memory-based language agent capable of emotional state adjustments, refusal detection, and personalized conversation flow.

### Features

- **Emotionally Aware Conversations**: Tracks the emotional state of the conversation using a two-dimensional emotion model.
- **Kolmogorov-Arnold Networks (KAN) Integration**: Enhances LLaMA's interaction by allowing KANs to adjust hidden states for improved conversational flow.
- **Refusal Detection and Override**: Identifies refusal phrases and adjusts the response using a specialized override mechanism.
- **Synthetic Day Cycle**: Mimics a day-night cycle to dynamically adjust the AI's behavior based on simulated time.
- **Live State Saving**: Captures and saves the model's state after each interaction for continuous learning.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Customization](#customization)
4. [How It Works](#how-it-works)
5. [Contributing](#contributing)
6. [Credits](#credits)
7. [License](#license)

## Installation

Follow these steps to set up **KAN-WuBu-Memory** on your system:

1. Clone the repository:

   ```bash
   git clone https://github.com/waefrebeorn/kan-wubu-memory.git
   ```

2. Navigate to the project directory:

   ```bash
   cd kan-wubu-memory
   ```

3. Run the setup script (`run.bat`) to initialize the environment and install dependencies:

   ```bash
   .\run.bat
   ```

4. Ensure that Python 3.8+ and CUDA-compatible drivers are installed. The script will automatically set up a virtual environment and install PyTorch, Hugging Face Transformers, and other dependencies.

5. **Important**: Manually download the required **LLaMA 3.2 1B** model files and place them in the `models/Llama_32_1B` directory.

   - You can download the files directly from Meta's LLaMA project or use Hugging Face's CLI.

6. Verify that the setup is complete:

   ```bash
   python -m llama_32_1b_tool --check
   ```

## Usage

Once the environment is set up, you can interact with the **KAN-WuBu-Memory** AI system through the GUI:

1. Launch the GUI:

   ```bash
   python main.py
   ```

2. In the GUI, enter your inputs, observe AI responses, and provide feedback to dynamically adjust the AI's emotional state.

3. The AI will adjust its responses based on emotional feedback, refusal detection, and conversation history. You can start new conversations or load saved states.

### Example Interaction

```
User: How are you feeling today?
AI: I feel quite neutral at the moment. How can I assist you?
```

The emotional state will shift dynamically based on the conversation context.

## Customization

You can adjust various components of the system to suit your needs:

- **System Prompt**: Customize the AI’s character description in `main.py` or directly through the GUI during the first interaction.
- **Emotional Feedback**: Modify the dimensions of emotional feedback to fit your use case (e.g., additional emotional dimensions like `confidence` or `interest`).
- **Synthetic Day Cycle**: Adjust the length and phases of the synthetic day cycle in `llama_32_1b_tool.py`.

## How It Works

### EmotionalState Module

The **EmotionalState** class tracks the AI’s emotional state in two dimensions (`pleasure` and `arousal`) and updates based on user feedback. This emotional model is used to generate more emotionally aware and context-sensitive responses.

### Refusal Detection

The **RefusalOverrideModule** monitors responses for refusal phrases (e.g., "I cannot assist with...") and attempts to override these using an RNN-based mechanism to ensure smoother interactions.

### Kolmogorov-Arnold Networks (KAN)

KANs are applied as a modification to LLaMA’s hidden layers, enabling the model to fine-tune and optimize conversational flow based on previous interactions. The **EnhancedKAN** class allows dynamic adjustments, resulting in a more personalized experience.

## Contributing

We welcome contributions from the community! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with detailed comments on your changes.

For major changes, please open an issue first to discuss what you would like to change.

## Credits

This project is built with contributions from various open-source libraries and developers. Special thanks to:

- **WuBu (WaefreBeorn)**: Project creator and lead developer.
- **Meta AI**: For the **LLaMA** language model that powers the core interaction.
- **Hugging Face**: For providing the **Transformers** library that makes working with modern NLP models accessible.
- **PyTorch Team**: For the foundational deep learning library that enables model training and optimization with CUDA support.
- **Contributors**: Open-source enthusiasts who provide libraries and frameworks like `matplotlib`, `scipy`, and more.

### Special Acknowledgments

- **LLaMA and Meta Research Team** for the original research behind the **LLaMA** language models.
- **Hugging Face Transformers Community** for their dedication to providing accessible NLP tools.
- **NVIDIA** for the CUDA toolkit, enabling efficient GPU computation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

