# KAN-WuBu-Memory: LLaMA 3.2 1B Instruct with Kolmogorov-Arnold Networks (KAN) Integration

![KAN-WuBu Memory](https://img.shields.io/badge/PyTorch-CUDA_Enabled-blue.svg)
![Version](https://img.shields.io/badge/Version-2.0.0-brightgreen.svg)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange.svg)

## Project Overview

**KAN-WuBu-Memory** is an advanced memory-integrated AI system that combines the **LLaMA 3.2 1B** language model with **Kolmogorov-Arnold Networks (KAN)** and a multi-dimensional memory framework. This system builds on the concept of emotionally aware, contextually sensitive, and dynamically evolving conversations. With support for long-term memory consolidation, real-time emotional modulation, and adaptive response generation, **KAN-WuBu-Memory** is designed for complex and nuanced conversational interactions.

### Key Features

- **Emotionally Aware Conversations**: Tracks and adjusts the AI's emotional state using a multi-dimensional model (`valence`, `arousal`, and `dominance`) to produce responses that align with contextual nuances.
- **Kolmogorov-Arnold Networks (KAN) Integration**: Enhances LLaMA’s interaction by leveraging KANs to adapt internal representations dynamically.
- **Advanced Memory Management**: Utilizes short-term, long-term, and sliding-window memories to retain context and adapt based on conversation history.
- **Refusal Detection and Override**: Identifies refusal phrases and uses corrective mechanisms to ensure smooth and continuous interaction.
- **Entropy-Based Response Management**: Uses entropy metrics to balance randomness and coherence in response generation.
- **Synthetic Day-Night Cycle**: Simulates a day-night cycle to influence the AI’s behavior dynamically, adjusting its responses and internal states based on simulated time.
- **Automatic State Saving and Loading**: Captures and saves the model’s state, memory, and emotional context after each interaction, allowing for continuous learning and persistent memory.
- **Dynamic Sampling Strategy**: Adjusts the sampling parameters (`temperature` and `top_p`) based on entropy, memory importance, and conversation context.

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

3. Run the setup script (`setup.bat` for Windows) to initialize the environment and install dependencies:

   ```bash
   .\setup.bat
   ```

4. Ensure that Python 3.8+ and CUDA-compatible drivers are installed. The script will automatically set up a virtual environment and install PyTorch, Hugging Face Transformers, and other dependencies.

5. **Important**: Manually download the required **LLaMA 3.2 1B** model files and place them in the `models/Llama_32_1B` directory.

   - You can download the files directly from Hugging Face's CLI by accepting the Llama License.

## Usage

Once the environment is set up, you can interact with the **KAN-WuBu-Memory** AI system:

**Kan GUI**: Start the interactive console mode:

   ```bash
   run run.bat
   ```


### Example Interaction

```
User: How are you feeling today?
AI: I feel quite neutral at the moment. How can I assist you?
```

The emotional state will shift dynamically based on the conversation context.

## Customization

You can adjust various components of the system to suit your needs:

- **System Prompt**: Customize the AI’s character description in `main.py` or directly through the GUI during the first interaction.
- **Emotional Feedback**: Modify the dimensions of emotional feedback to fit your use case (e.g., add `confidence`, `interest`).
- **Synthetic Day Cycle**: Adjust the length and phases of the synthetic day cycle in `llama_32_1b_tool.py`.
- **Memory Management**: Configure short-term and long-term memory buffers, and adjust the clustering for memory consolidation.
- **Entropy Management**: Change entropy thresholds and sampling parameters (`temperature`, `top_p`) for response generation.

## How It Works

### EmotionalState Module

The **EmotionalState** class tracks the AI’s emotional state across three dimensions (`valence`, `arousal`, and `dominance`) and updates based on user feedback and conversation context. This emotional model is used to generate emotionally aware and context-sensitive responses.

### Overfit Detector

The **OverfitDetector** monitors loss trends across training and validation windows to identify signs of overfitting and trigger adjustments, such as early stopping or dynamic learning rate scaling.

### Kolmogorov-Arnold Networks (KAN)

KANs modify the hidden layers of LLaMA, allowing the system to fine-tune and optimize its internal representations based on emotional and contextual inputs. The **EnhancedKAN** class enables dynamic adjustments, resulting in a more personalized conversational experience.

### Refusal Detection and Override

The **RefusalDetector** module monitors for refusal phrases (e.g., "I cannot assist with...") and utilizes a KAN-powered override to refine and rephrase these responses, ensuring a smoother interaction flow.

### Memory Management

The **AdvancedMemoryManager** handles multi-dimensional memory, integrating short-term, long-term, and sliding-window memories to consolidate and prioritize context. This module supports clustering, importance scoring, and context summarization for efficient memory management.

### Entropy-Based Response Quality Management

The **EntropyManager** tracks the entropy of generated responses, ensuring a balance between coherence and randomness. Entropy metrics are used to adjust sampling parameters (`temperature`, `top_p`), and trigger "chain-of-thought" reasoning processes when necessary.

### Synthetic Day-Night Cycle

The **SyntheticDayCycle** simulates a day-night cycle that influences the AI’s internal state. The cycle affects behavior, response length, and sampling parameters based on the time of day.

### Live State Saving

After each interaction, the system captures and saves the current state (including emotional context, memory buffers, and learning metrics) to ensure continuous learning and persistence.

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

