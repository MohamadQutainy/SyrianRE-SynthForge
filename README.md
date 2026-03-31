# Syrian Real Estate Synthetic Data Generator 🏘️🤖

A sophisticated multi-model framework designed to generate high-quality, structured synthetic data for the Syrian real estate market. This tool bridges the gap in localized data scarcity, enabling the development of AI models tailored for the Syrian context.

## 🚀 Overview
In many emerging markets, real-world data for training Machine Learning models is either private, fragmented, or unstructured. This project provides a scalable solution by leveraging Large Language Models (LLMs) to generate realistic, diverse, and structured real estate listings (Apartments, Land, Commercial) specifically for Syrian cities like Damascus, Aleppo, Homs, and more.

## ✨ Key Features
- **Multi-Model Support:** Seamless integration with **OpenAI (GPT-4o/o1)**, **Anthropic (Claude 3.5)**, and local deployment via **HuggingFace/Ollama (Llama 3.2-3B)**.
- **Structured Output:** Guaranteed JSON, CSV, and XLSX formats for easy integration into SQL/NoSQL databases or downstream RAG pipelines.
- **Localized Context:** Includes deep knowledge of Syrian neighborhoods (e.g., Malki, Kafr Souseh, Dummar), local currency fluctuations (SYP/USD exchange rates), and regional property features.
- **Dynamic Configuration:** Extensible pools for cities, property types, features, and price ranges.
- **Interactive UI:** Professional **Gradio** interface for real-time data generation, filtering, and preview.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **LLM Orchestration:** OpenAI API, Anthropic SDK, HuggingFace Transformers.
- **Data Handling:** Pandas, Openpyxl, JSON.
- **Frontend/UI:** Gradio.
- **Environment Management:** Python-dotenv.

## 🏗️ Architecture
The system follows a modular design:
1. **Reference Layer (`config.py`):** Static configuration for Syrian geography, market metrics, and model definitions.
2. **Logic Layer (`models_logic.py`):** Core engine for prompt engineering and LLM API handling.
3. **UI Layer (`gradio_ui.py`):** Encapsulates the frontend logic and user interactions.
4. **Entry Point (`app.py`):** The main script to launch the application.

## 📋 Prerequisites
- Python 3.9 or higher.
- API Keys for OpenAI or Anthropic (Optional if running local models).
- GPU (Optional, required for running Llama locally via Transformers).

## 🚀 Setup & Installation

Follow these steps to get the project running locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/Syrian-Real-Estate-Data-Generator.git](https://github.com/YourUsername/Syrian-Real-Estate-Data-Generator.git)
   cd Syrian-Real-Estate-Data-Generator