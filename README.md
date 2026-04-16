# IMDB Sentiment Analysis — End-to-End ML Deployment on AWS

![Python](https://img.shields.io/badge/Python-3.6+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?logo=pytorch)
![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Lambda%20%7C%20API%20Gateway-FF9900?logo=amazonaws)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Overview

A production-style sentiment analysis system that classifies IMDB movie reviews as positive or negative through a publicly accessible web application. The project covers the full ML lifecycle — data preprocessing, model training, cloud deployment, and serving predictions via a REST API — all orchestrated on AWS.

This isn't just a model in a notebook. It's a deployed, end-to-end inference pipeline where a user types a review into a web page and gets a real-time prediction from a model running on SageMaker.

## ✨ Key Features

- **Custom LSTM model** built from scratch in PyTorch for binary sentiment classification
- **Full SageMaker integration** — training, model artifact management, and endpoint deployment
- **Custom inference pipeline** — raw text input is preprocessed server-side (HTML stripping, stemming, stopword removal, vocabulary encoding) before reaching the model
- **Serverless API layer** — Lambda + API Gateway expose the SageMaker endpoint as a public REST API, decoupling the frontend from AWS credentials
- **Lightweight web frontend** that communicates with the model via a single XHR POST request

## 🧠 Technical Highlights

- **Separation of training and serving code**: The `train/` and `serve/` directories isolate training logic from inference logic, each with their own dependencies and entry points — mirroring production ML repo structure
- **Custom SageMaker inference handlers**: Implements all four SageMaker serving contract functions (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) to accept raw text instead of pre-processed tensors
- **Text preprocessing pipeline**: Multi-step NLP pipeline (HTML parsing → regex cleaning → stopword removal → Porter stemming → vocabulary lookup → fixed-length padding) ensures consistent transformation between training and inference
- **Vocabulary-based encoding with OOV handling**: Words outside the top 4,998 are mapped to a single "infrequent" token (index 1), and padding uses index 0 — a deliberate design to keep the embedding matrix small while handling unseen words gracefully
- **Variable-length review handling**: Reviews are padded/truncated to 500 tokens, with actual length passed as the first element of the input tensor so the LSTM output is extracted at the correct timestep

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch (nn.Module, LSTM, Embedding) |
| Cloud ML Platform | Amazon SageMaker (Training, Endpoints) |
| Serverless Compute | AWS Lambda |
| API | Amazon API Gateway (REST, Lambda Proxy) |
| NLP | NLTK (stopwords, Porter Stemmer), BeautifulSoup |
| Frontend | HTML, Bootstrap 3, vanilla JavaScript (XHR) |
| Data | IMDB Dataset (50,000 reviews) |

## 🏗 Architecture / How It Works

```
┌──────────┐     POST (raw text)     ┌─────────────┐     invoke      ┌───────────────────┐
│  Browser  │ ──────────────────────► │ API Gateway  │ ──────────────► │  Lambda Function   │
│ (HTML/JS) │ ◄────────────────────── │  (REST API)  │ ◄────────────── │  (boto3 runtime)   │
└──────────┘     "0" or "1"          └─────────────┘    response     └────────┬──────────┘
                                                                              │
                                                                   invoke_endpoint()
                                                                              │
                                                                              ▼
                                                                   ┌──────────────────┐
                                                                   │ SageMaker Endpoint│
                                                                   │                  │
                                                                   │  input_fn()      │
                                                                   │  → deserialize   │
                                                                   │  predict_fn()    │
                                                                   │  → preprocess    │
                                                                   │  → LSTM forward  │
                                                                   │  output_fn()     │
                                                                   │  → serialize     │
                                                                   └──────────────────┘
```

**Data flow during inference:**
1. User submits a raw movie review string via the web form
2. API Gateway forwards the request body to a Lambda function
3. Lambda invokes the SageMaker endpoint using `boto3`
4. The endpoint's `input_fn` deserializes the text, `predict_fn` preprocesses it (clean → tokenize → stem → encode → pad to 500 → tensor), runs the LSTM forward pass, and rounds the sigmoid output to 0 or 1
5. The result propagates back through Lambda → API Gateway → browser, which displays "POSITIVE" or "NEGATIVE"

**Model architecture:**
```
Input [1 + 500] → Embedding(5000, 32) → LSTM(32, 100) → Linear(100, 1) → Sigmoid → {0, 1}
```

## ⚡ Getting Started

### Prerequisites

- AWS Account with SageMaker, Lambda, and API Gateway access
- Python 3.6+
- Jupyter Notebook (or SageMaker Notebook Instance)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/jashjain21/IMDB-sentiments-on-AWS.git
   ```

2. **Launch the SageMaker notebook**
   - Create a SageMaker Notebook Instance
   - Upload `SageMaker Project.ipynb` and the `train/` and `serve/` directories
   - Run all cells sequentially — the notebook handles data download, preprocessing, training, and deployment

3. **Set up the serverless API** (after the endpoint is deployed)
   - Create an IAM role with `AmazonSageMakerFullAccess` for Lambda
   - Create a Lambda function with the `boto3` invocation code (provided in the notebook)
   - Create an API Gateway REST API with a POST method pointing to the Lambda function
   - Deploy the API to a stage (e.g., `prod`)

4. **Deploy the web app**
   - Update the API Gateway URL in `website/index.html` (the `form action` attribute)
   - Open `index.html` in a browser

> ⚠️ **Cost note**: The SageMaker endpoint incurs charges while running. Shut it down via `predictor.delete_endpoint()` when not in use.

## 📌 Example Usage

Open the web app, paste a review, and click Submit:

| Input | Output |
|---|---|
| *"This movie was absolutely wonderful. The acting was superb and the story kept me engaged throughout."* | ✅ **POSITIVE** |
| *"Terrible film. The plot made no sense and the dialogue was painful to sit through."* | ❌ **NEGATIVE** |

## 🔍 What This Project Demonstrates

- **End-to-end ML engineering** — from raw data to a user-facing application
- **AWS cloud architecture** — SageMaker, Lambda, API Gateway working together as a serverless inference pipeline
- **PyTorch model development** — custom LSTM implementation with embedding layers
- **NLP preprocessing** — tokenization, stemming, stopword removal, vocabulary encoding, sequence padding
- **Custom SageMaker inference code** — writing serving handlers that accept raw text instead of pre-processed tensors
- **Serverless API design** — using Lambda as a bridge between a public API and a private ML endpoint
- **ML deployment best practices** — separating training/serving code, artifact management, cost-aware endpoint lifecycle

## 🚧 Limitations / Future Improvements

- **No CI/CD pipeline** — training and deployment are manual via the notebook
- **Single-instance endpoint** — no auto-scaling configured for the SageMaker endpoint
- **No model versioning** — retraining overwrites the previous model with no A/B testing or rollback
- **Basic frontend** — no input validation, error handling, or loading states in the web app
- **Fixed vocabulary** — the word dictionary is built once at training time; new slang or terminology won't be recognized
- **No authentication** — the API Gateway endpoint is publicly accessible with no rate limiting
- **Batch transform not used for evaluation** — test inference is done one review at a time, which is slow

**Potential improvements:**
- Add CloudFormation/CDK templates for infrastructure-as-code deployment
- Implement auto-scaling on the SageMaker endpoint
- Add a model registry (SageMaker Model Registry) for versioning
- Replace the LSTM with a fine-tuned transformer (e.g., DistilBERT) for better accuracy
- Add CloudWatch monitoring and alarms for endpoint latency/errors

## 📎 Why This Matters (For Recruiters)

- **Demonstrates full-stack ML deployment** — not just model training, but building the entire cloud infrastructure to serve predictions to end users
- **Shows AWS proficiency** — hands-on use of SageMaker, Lambda, API Gateway, S3, and IAM in a cohesive architecture
- **Proves ability to write production-style inference code** — custom SageMaker handlers that bridge the gap between raw user input and model expectations
- **Exhibits understanding of system design trade-offs** — serverless API layer, cost management (endpoint lifecycle), and separation of concerns between training and serving
- **Covers the ML lifecycle end-to-end** — data collection, preprocessing, training, evaluation, deployment, and monitoring considerations
