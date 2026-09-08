# 🧠 Azure Sentiment Analyser

A Python CLI tool that uses **Microsoft Azure Cognitive Services (Text Analytics)** to perform real-time sentiment analysis, key phrase extraction, and language detection on any text — one line or a whole file.

Built to demonstrate practical application of the **Microsoft Azure AI Fundamentals (AI-900)** certification.

---

## Features

- 😊 **Sentiment analysis** — Positive / Neutral / Negative / Mixed with confidence scores
- 📊 **Visual sentiment bar** — colour-coded breakdown at a glance
- 🔑 **Key phrase extraction** — identifies the most important terms
- 🌍 **Language detection** — detects language with confidence score
- 🔍 **Opinion mining** — aspect-based sentiment (e.g. "battery is bad")
- 📄 **Batch file processing** — analyse hundreds of texts from a `.txt` file
- 💾 **JSON export** — save full results for further analysis
- 🖥️ **Interactive mode** — type and analyse in real time

---

## Tech Stack

| Component    | Technology                           |
|--------------|--------------------------------------|
| Language     | Python 3.10+                         |
| Cloud API    | Azure Cognitive Services (AI-900)    |
| SDK          | `azure-ai-textanalytics`             |
| Auth         | Azure Key Credential                 |

---

## Prerequisites

- A Microsoft Azure account
- An Azure **Language** resource (formerly Text Analytics)
- Python 3.10+

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ekql-ops/azure-sentiment.git
cd azure-sentiment-analyser
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Azure credentials

Create a `.env` file in the root directory:

```env
AZURE_LANGUAGE_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com
AZURE_LANGUAGE_KEY=your_api_key_here
```

> Get these from the Azure Portal → Your Language resource → Keys and Endpoint.

---

## Usage

### Interactive mode (type and analyse)

```bash
python src/analyser.py
```

### Analyse a single text

```bash
python src/analyser.py -t "The food was absolutely amazing but the service was terrible."
```

### Analyse from a file (one text per line)

```bash
python src/analyser.py -f reviews.txt
```

### Save results as JSON

```bash
python src/analyser.py -f reviews.txt --save
```

---

## Example Output

```
  ╔══════════════════════════════════════╗
  ║   Azure Sentiment Analyser v1.0      ║
  ╚══════════════════════════════════════╝
  [✓] Connected to Azure Text Analytics.

  ┌─ Result 1 ─────────────────────────────────────────
  │  Text       : The food was amazing but service was terrible.
  │  Language   : English (en) — 100% confidence
  │  Sentiment  : 🤔  MIXED
  │  Bar        : [████████████████░░░░░░░░████████]
  │             :  +  52.3%  ~   5.1%  -  42.6%
  │  Key Phrases: food, service
  │
  │  Sentence breakdown:
  │    😊  The food was amazing
  │       ↳ food: amazing (positive)
  │    😞  but the service was terrible.
  │       ↳ service: terrible (negative)
  └─────────────────────────────────────────────────────
```

---

## File Structure

```
azure-sentiment-analyser/
├── src/
│   └── analyser.py      # Main application
├── samples/
│   └── reviews.txt      # Sample texts to test with
├── .env.example         # Template for credentials
├── requirements.txt
└── README.md
```

---

## Requirements

```
azure-ai-textanalytics>=5.3.0
python-dotenv>=1.0.0
```

---

## Related Certifications

This project demonstrates skills from:
- **Microsoft Azure AI Fundamentals (AI-900)**
- **Microsoft Azure Fundamentals (AZ-900)**

---

## Author

**Abubakar Albakri** — BSc Computer Science, Manchester Metropolitan University  
[GitHub](https://github.com/ekql-ops)
