# Vera Bot — magicpin AI Challenge

A high-performance merchant-AI assistant for magicpin's "Vera" product. This bot handles proactive merchant engagement and automated customer communication with a hybrid deterministic-LLM architecture.

## 🚀 Live API Access
The bot is deployed and accessible at:
**`verabot-production-fffb.up.railway.app`**

---

## 🛠 Key Features

- **3-Tier Intent Classification**: Routes merchant replies using a strict hierarchy:
  1. **Regex Fast-Path**: Instant matching for common patterns.
  2. **Semantic Similarity (BGE-M3)**: Cosine similarity via Hugging Face Inference for high-precision intent detection.
  3. **LLM Reasoning (Gemma-3)**: Zero-temperature fallback for complex or ambiguous cases.
- **Deterministic Composer**: Ensures 100% reliability with Pydantic validation, voice enforcement, and fact-checking guards.
- **Smart Load Balancing**: Automatic failover across multiple Gemini API keys with per-key cooldowns and model-chain fallback.
- **Stateless & Scalable**: Designed to meet strict judge harness SLAs (<30s response time).

## 📁 Endpoints

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/v1/context` | POST | Load category, merchant, customer, or trigger data. |
| `/v1/tick` | POST | Generate proactive engagement actions. |
| `/v1/reply` | POST | Respond to merchant/customer messages dynamically. |
| `/v1/healthz` | GET | Liveness probe and context status. |
| `/v1/metadata` | GET | Bot identity and architectural summary. |
| `/v1/teardown` | POST | Wipe in-memory state for a clean test reset. |

## 💻 Local Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file with:
   - `GEMINI_API_KEY`: For message composition and intent reasoning.
   - `HF_TOKEN`: For semantic similarity matching.

3. **Run the Bot**:
   ```bash
   uvicorn bot:app --host 0.0.0.0 --port 8080
   ```

## 🧪 Testing

- **Unit Tests**: `python -m unittest discover -s tests`
- **Judge Simulator**: `python tests/judge_simulator.py`
- **End-to-End Harness**: `python conversationTest.py`
