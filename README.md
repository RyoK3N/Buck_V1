# Buck_V1
An agent that helps you predict the next day's stock data.

# 🤖 BUCK (Version 1.4.1) - AI-Powered Stock Analysis & Prediction

A production-ready stock analysis and prediction system that combines technical analysis, sentiment analysis, and AI-powered forecasting using OpenAI models.

## ✨ Features

- **📊 Comprehensive Technical Analysis**: 6+ technical indicators (RSI, MACD, Moving Averages, OBV, Support/Resistance, Candlestick Patterns)
- **📰 Sentiment Analysis**: News sentiment analysis using keyword-based scoring
- **🤖 AI-Powered Predictions**: OpenAI GPT-4 powered next-day price forecasting
- **⚡ Async & Concurrent**: High-performance async architecture with concurrent processing
- **🔧 Production Ready**: FAANG-level software engineering practices with SOLID principles
- **📈 Batch Analysis**: Analyze multiple stocks simultaneously
- **💾 Caching & Storage**: Built-in result caching and file storage
- **🛠️ Extensible**: Modular design with factory patterns and dependency injection

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RyoK3N/Buck_V1
cd Buck_V1

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export INDIAN_API_KEY="your-indian-api-key"  # Optional for news data

# Or configure keys once using Python (creates a .env file)
python - <<'EOF'
from agent_scripts import set_api_keys
set_api_keys("your-openai-api-key", "your-indian-api-key")
EOF
```

### Basic Usage

```python
import asyncio
from agent_scripts import create_agent

async def analyze_stock_example():
    # Create the agent
    agent = create_agent(
        openai_api_key="your-openai-api-key",
        indian_api_key="your-indian-api-key",  # Optional
        model="gpt-4o"
    )
    
    # Analyze a stock
    async with agent:
        results = await agent.analyze_and_predict(
            symbol="BHEL.NS",
            start_date="2024-01-01", 
            end_date="2024-01-10",
            interval="1h"
        )
    
    # Display forecast
    forecast = results['forecast']
    print(f"Next-day forecast for BHEL.NS:")
    print(f"Close: ₹{forecast['close']:.2f}")
    print(f"Confidence: {forecast['confidence']:.2f}")
    print(f"Reasoning: {forecast['reasoning']}")

# Run the example
asyncio.run(analyze_stock_example())
```

### Command Line Interface

```bash
# Analyze a single stock
python -m agent_scripts.cli analyze BHEL.NS --start 2025-06-18 --end 2024-06-20

# Run demo
python -m agent_scripts.cli demo --symbol BHEL.NS

# Use custom parameters
python -m agent_scripts.cli analyze BHEL.NS \
  --start 2024-01-01 \
  --end 2024-01-10 \
  --interval 30m \
  --model gpt-4o \
  --api-key your-openai-key
```

## 📖 API Documentation

### StockAgent

The main orchestrator class that coordinates data collection, analysis, and prediction.

```python
from agent_scripts import BuckFactory

# Create with default settings
agent = BuckFactory.create_default_agent()

# Create with custom configuration
agent = BuckFactory.create_production_agent(
    openai_api_key="your-key",
    indian_api_key="your-news-key",
    model="gpt-4o"
)
```

#### Key Methods

- `analyze_and_predict()`: Complete analysis workflow for a single stock
- `batch_analyze()`: Analyze multiple stocks concurrently
- `get_cached_analysis()`: Retrieve cached analysis results
- `clear_cache()`: Clear all cached data

### Analysis Results Structure

```python
{
    "symbol": "BHEL.NS",
    "timestamp": "2024-01-15T10:30:00",
    "data_info": {
        "start_date": "2024-01-01",
        "end_date": "2024-01-10", 
        "interval": "1h",
        "data_points": 168,
        "news_available": True
    },
    "analysis_results": [...],  # Technical and sentiment analysis
    "forecast": {
        "date": "2024-01-11",
        "open": 245.30,
        "high": 248.75,
        "low": 242.10,
        "close": 247.80,
        "confidence": 0.75,
        "reasoning": "Technical indicators show bullish momentum..."
    },
    "metadata": {
        "agent_version": "1.0.0",
        "model_used": "gpt-4o",
        "analysis_confidence": 0.82,
        "prediction_confidence": 0.75
    }
}
```

## 🔧 Architecture

The system follows SOLID principles with a clean, modular architecture:

```
market_forecaster/
├── interfaces.py          # Abstract interfaces and protocols
├── stock_agent.py         # Main orchestrator 
├── data_providers.py      # Data collection (Yahoo Finance, News API)
├── tools.py              # Technical analysis tools
├── analyzers.py          # Analysis coordinators  
├── predictors.py         # AI-powered prediction
├── config.py             # Configuration management
└── cli.py                # Command-line interface
```

### Key Components

1. **Data Providers**: Fetch stock data (Yahoo Finance) and news data (Indian API)
2. **Analysis Tools**: Individual technical indicators (RSI, MACD, etc.)
3. **Analyzers**: Coordinate multiple tools for comprehensive analysis
4. **Predictors**: Generate AI-powered forecasts using OpenAI
5. **Stock Agent**: Main orchestrator that ties everything together

## 🛠️ Technical Indicators

| Indicator | Purpose | Signals |
|-----------|---------|---------|
| **RSI** | Overbought/Oversold conditions | Buy < 30, Sell > 70 |
| **MACD** | Trend momentum and direction | Buy when MACD > Signal |
| **Moving Averages** | Trend identification | Buy when price > MA |
| **OBV** | Volume flow analysis | Rising OBV = Buying pressure |
| **Support/Resistance** | Key price levels | Buy near support, sell near resistance |
| **Candlestick Patterns** | Reversal signals | 19 classical patterns detected |

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root to store these variables. The
`.gitignore` file already excludes `.env` so your secrets remain private.

You can generate this file automatically using `set_api_keys`:

```python
from agent_scripts import set_api_keys
set_api_keys("your-openai-key", "your-indian-key")
```

```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional
INDIAN_API_KEY=your-indian-api-key
CHAT_MODEL=gpt-4o
TEMPERATURE=0.1
MAX_COMPLETION_TOKENS=500
NEWS_ITEMS=10
LOG_LEVEL=INFO
OUTPUT_DIR=output
```

### Custom Configuration

```python
from agent_scripts.interfaces import BuckConfig
from agent_scripts import BuckFactory

config = BuckConfig(
    openai_api_key="your-key",
    chat_model="gpt-4o",
    temperature=0.1,
    max_tokens=500,
    news_items=10,
    log_level="INFO"
)

agent = BuckFactory.create_custom_agent(config)
```

## 📊 Performance Features

- **Async Architecture**: Non-blocking I/O for better performance
- **Concurrent Processing**: Analyze multiple stocks simultaneously
- **Caching**: Built-in result caching to avoid redundant API calls
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Rate Limiting**: Built-in concurrency controls to respect API limits

## 🧪 Testing

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=market_forecaster --cov-report=html
```

## 📈 Example Output

```
🔍 Analyzing BHEL.NS from 2024-01-01 to 2024-01-10...

📈 Analysis Results for BHEL.NS
==================================================
Data Period: 2024-01-01 to 2024-01-10 (1h)
Data Points: 168
News Available: Yes
Model Used: gpt-4o

🔮 Next-Day Forecast (2024-01-11)
------------------------------
Open:  ₹245.30
High:  ₹248.75
Low:   ₹242.10
Close: ₹247.80
Confidence: 0.75
Reasoning: Technical indicators show bullish momentum with RSI at 65, MACD positive crossover, and price above key moving averages. News sentiment is neutral with no major negative factors.

📊 Confidence Scores
--------------------
Analysis: 0.82
Prediction: 0.75

💾 Results saved to output directory
```

## 🚀 Production Deployment

### Environment Setup

```bash
# Production environment variables
export OPENAI_API_KEY="prod-openai-key"
export INDIAN_API_KEY="prod-indian-key"
export LOG_LEVEL="INFO"
export CHAT_MODEL="gpt-4o"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Links

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Technical Analysis Indicators](https://www.investopedia.com/terms/t/technicalanalysis.asp)

---

** Now lets start making some Bucks 💰💰💰 ** 
