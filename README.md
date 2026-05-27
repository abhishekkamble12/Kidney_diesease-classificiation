# AI-Native Sentiment Intelligence Platform

A production-grade, full-stack AI analytics platform designed to classify public feedback sentiments and run detailed LangChain-driven search and report analysis.

## Architecture

This project provides a robust backend paired with an enterprise-grade, lightweight frontend. It uses a multi-tiered approach to ensure minimal latency, utilizing local ML models (LinearSVC, DistilBERT) in an ensemble setup to guarantee accurate predictions without relying on third-party cloud APIs.

### Frontend
- **Tech Stack**: FastAPI Templates, Jinja2, Tailwind CSS, Alpine.js, Chart.js, ECharts, HTMX.
- **Design Language**: Dark theme, glassmorphism, responsive, and highly interactive.
- **Key Modules**:
  - **Predict Dashboard**: Real-time sentiment prediction with ChatGPT-style explanations.
  - **Search Analysis**: AI-powered dynamic search aggregation and sentiment analysis.
  - **Bulk CSV Analysis**: Drag-and-drop file processing with immediate charting insights.
  - **Monitoring & Analytics**: Enterprise-level metrics visualization (latencies, confidence drift, model fallback rates).

### Backend
- **Langchain Search & Reports**: Utilizes DuckDuckGo (and RSS) for high-availability searching, aggregating public feedback dynamically.
- **PostgreSQL Persistence**: Captures complete system observability metrics (latency, fallback indicators, inference distribution).
- **MLflow & DagsHub**: Fully integrated for experiment tracking and model monitoring.

## Run Instructions

### Prerequisites
- Python 3.10+ or Docker.
- A valid PostgreSQL database connection URL.
- API Keys: LangChain, Groq, and Hugging Face.

### Running Locally (Python)

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Environment Configuration**
Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/tweet_sentiment
GROQ_API_KEY=your_groq_api_key
HF_API_TOKEN=your_hugging_face_token
```

3. **Start the Platform**
```bash
python app.py
```
*The platform UI and API will be available at http://localhost:8000*

### Running via Docker
The provided `Dockerfile` creates a highly optimized layer capable of executing the entire multi-model pipeline.
```bash
docker build -t tweet-sentiment-platform .
docker run -d -p 8000:8000 --env-file .env tweet-sentiment-platform
```

## Dependencies Overview
- **FastAPI / Uvicorn**: High-performance asynchronous API framework and server, alongside Jinja2 templates.
- **Transformers / Scikit-Learn**: Hybrid ML pipeline providing sub-100ms inference locally.
- **Langchain**: Agent orchestration and high-speed search aggregation.
- **SQLAlchemy / Alembic**: Database abstraction and schema migration support.
- **TailwindCSS / Alpine.js**: Lightweight styling and frontend reactivity.