"""
Buck_V1.predictors
──────────────────────────────
AI-powered stock prediction implementations.
"""

from __future__ import annotations
import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import openai
from openai import OpenAI, OpenAIError

from .interfaces import IPredictor, AnalysisResult, Forecast
from .config import SETTINGS, LOGGER


class OpenAIPredictor(IPredictor):
    """OpenAI-powered stock predictor."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.1):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self._json_pattern = re.compile(r"\{.*\}", re.DOTALL)
    
    async def predict(
        self,
        analysis_results: List[AnalysisResult],
        **kwargs
    ) -> Forecast:
        """Generate stock forecast using OpenAI."""
        try:
            if not analysis_results:
                raise ValueError("No analysis results provided")
            
            symbol = analysis_results[0]['symbol']
            LOGGER.info(f"Generating forecast for {symbol} using OpenAI")
            
            # Build prompts from analysis results
            system_prompt, user_prompt = self._build_prompts(analysis_results, **kwargs)
            
            # Call OpenAI API
            forecast_data = await self._call_openai_async(system_prompt, user_prompt)
            
            # Validate and format response
            forecast = self._validate_and_format_forecast(forecast_data, symbol)
            
            LOGGER.info(f"Successfully generated forecast for {symbol}")
            return forecast
            
        except Exception as e:
            LOGGER.error(f"Prediction failed for {symbol}: {e}")
            # Return a minimal forecast with error information
            return Forecast(
                date=datetime.now().strftime('%Y-%m-%d'),
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
                confidence=0.0,
                reasoning=f"Prediction failed: {str(e)}"
            )
    
    async def _call_openai_async(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Make async OpenAI API call."""
        try:
            # Run the synchronous OpenAI call in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_openai_sync,
                system_prompt,
                user_prompt
            )
            return response
        except Exception as e:
            LOGGER.error(f"OpenAI API call failed: {e}")
            raise
    
    def _call_openai_sync(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Synchronous OpenAI API call."""
        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": self.temperature,
            }
            
            # Add token limits based on model
            if self.model.lower().startswith("o1") or "gpt-4o" in self.model.lower():
                kwargs["max_completion_tokens"] = SETTINGS.max_completion_tokens
            else:
                kwargs["max_tokens"] = SETTINGS.max_completion_tokens
            
            LOGGER.debug(f"Calling OpenAI API with model: {self.model}")
            response = self.client.chat.completions.create(**kwargs)
            
            raw_content = response.choices[0].message.content or ""
            
            # Parse JSON response
            try:
                return json.loads(raw_content)
            except json.JSONDecodeError:
                # Fallback: extract JSON using regex
                match = self._json_pattern.search(raw_content)
                if match:
                    return json.loads(match.group(0))
                else:
                    raise ValueError(f"Could not parse JSON from response: {raw_content}")
                    
        except Exception as e:
            LOGGER.error(f"OpenAI sync call failed: {e}")
            raise
    
    def _build_prompts(self, analysis_results: List[AnalysisResult], **kwargs) -> tuple[str, str]:
        """Build system and user prompts from analysis results."""
        symbol = analysis_results[0]['symbol']
        
        # System prompt
        system_prompt = """You are a senior quantitative analyst and portfolio manager with 15+ years of experience in equity research and algorithmic trading.

Your task is to generate a precise next-day stock price forecast based on comprehensive technical and sentiment analysis.

Return ONLY a valid JSON object with these exact keys:
- "date": Next trading day in YYYY-MM-DD format
- "open": Opening price prediction (float, 2 decimals)
- "high": High price prediction (float, 2 decimals) 
- "low": Low price prediction (float, 2 decimals)
- "close": Closing price prediction (float, 2 decimals)
- "confidence": Confidence score 0.0-1.0 (float, 2 decimals)
- "reasoning": Concise analysis reasoning (string, max 100 words)

Guidelines:
1. Base predictions on technical indicators (RSI, MACD, moving averages)
2. Consider volume patterns (OBV) and candlestick signals
3. Factor in support/resistance levels and sentiment analysis
4. Ensure high >= max(open, close) and low <= min(open, close)
5. Lower confidence if indicators conflict or data is sparse
6. Be conservative with price movements (typically 0.5-3% daily)

Output only the JSON object, no additional text."""

        # User prompt - aggregate analysis data
        user_prompt = self._format_analysis_data(analysis_results)
        
        return system_prompt, user_prompt
    
    def _safe_format_float(self, value, decimals=2):
        """Safely format a value that might be 'N/A' or a number."""
        if value == 'N/A' or value is None:
            return 'N/A'
        try:
            return f"{float(value):.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)
    
    def _format_analysis_data(self, analysis_results: List[AnalysisResult]) -> str:
        """Format analysis results into a comprehensive prompt."""
        symbol = analysis_results[0]['symbol']
        
        # Initialize data sections
        technical_data = {}
        sentiment_data = {}
        current_price = None
        
        # Process each analysis result
        for result in analysis_results:
            if result['analysis_type'] == 'technical_analysis':
                technical_data = result['data']
            elif result['analysis_type'] == 'sentiment_analysis':
                sentiment_data = result['data']
            elif result['analysis_type'] == 'composite_analysis':
                # Extract nested technical and sentiment data
                nested_results = result['data'].get('analysis_results', {})
                if 'technical_analysis' in nested_results:
                    technical_data = nested_results['technical_analysis'].get('data', {})
                if 'sentiment_analysis' in nested_results:
                    sentiment_data = nested_results['sentiment_analysis'].get('data', {})
        
        # Extract key technical metrics
        summary = technical_data.get('summary', {})
        key_metrics = summary.get('key_metrics', {})
        
        # Format technical indicators
        technical_section = f"""### Technical Analysis for {symbol}
Overall Signal: {summary.get('overall_signal', 'N/A')} (Strength: {self._safe_format_float(summary.get('signal_strength', 0))})

Key Indicators:
- RSI: {self._safe_format_float(key_metrics.get('rsi', 'N/A'))} ({key_metrics.get('rsi_condition', 'N/A')})
- MACD: {self._safe_format_float(key_metrics.get('macd', 'N/A'), 4)} | Signal: {self._safe_format_float(key_metrics.get('macd_signal', 'N/A'), 4)}
- Moving Averages: Short MA: {self._safe_format_float(key_metrics.get('short_ma', 'N/A'))} | Long MA: {self._safe_format_float(key_metrics.get('long_ma', 'N/A'))}
- Support: {key_metrics.get('nearest_support', 'N/A')} | Resistance: {key_metrics.get('nearest_resistance', 'N/A')}
- Candlestick Patterns: {key_metrics.get('pattern_count', 0)} detected
  - Bullish Score: {self._safe_format_float(key_metrics.get('bullish_score', 0), 1)}
  - Bearish Score: {self._safe_format_float(key_metrics.get('bearish_score', 0), 1)}

Signal Breakdown: {summary.get('signals_breakdown', {})}
Tools Processed: {summary.get('tools_processed', 0)}/{summary.get('tools_processed', 0) + summary.get('tools_failed', 0)}"""

        # Format sentiment analysis
        sentiment_section = f"""### Sentiment Analysis
News Available: {sentiment_data.get('news_available', False)}
News Count: {sentiment_data.get('news_count', 0)}
Sentiment: {sentiment_data.get('sentiment_label', 'NEUTRAL')} (Score: {self._safe_format_float(sentiment_data.get('sentiment_score', 0))})
Positive Mentions: {sentiment_data.get('positive_mentions', 0)}
Negative Mentions: {sentiment_data.get('negative_mentions', 0)}"""

        # Get data info
        data_info = technical_data.get('data_info', {})
        data_section = f"""### Data Information
Period: {data_info.get('start_date', 'N/A')} to {data_info.get('end_date', 'N/A')}
Interval: {data_info.get('interval', 'N/A')}
Data Points: {data_info.get('data_points', 'N/A')}"""

        # Try to extract current price from tools results
        tools_results = technical_data.get('tools_results', {})
        for tool_name, result in tools_results.items():
            if 'current_price' in result:
                current_price = result['current_price']
                break
        
        if current_price:
            price_section = f"Current Price: ₹{self._safe_format_float(current_price)}"
        else:
            price_section = "Current Price: Not available"

        return f"""{data_section}

{price_section}

{technical_section}

{sentiment_section}

Please provide your next-day forecast based on this comprehensive analysis."""
    
    def _validate_and_format_forecast(self, forecast_data: Dict[str, Any], symbol: str) -> Forecast:
        """Validate and format the forecast response."""
        required_fields = {'date', 'open', 'high', 'low', 'close', 'confidence', 'reasoning'}
        
        # Check required fields
        missing_fields = required_fields - set(forecast_data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate price relationships
        open_price = float(forecast_data['open'])
        high_price = float(forecast_data['high'])
        low_price = float(forecast_data['low'])
        close_price = float(forecast_data['close'])
        
        if high_price < max(open_price, close_price):
            LOGGER.warning("Adjusting high price to maintain price relationships")
            high_price = max(open_price, close_price)
        
        if low_price > min(open_price, close_price):
            LOGGER.warning("Adjusting low price to maintain price relationships")
            low_price = min(open_price, close_price)
        
        # Validate confidence
        confidence = max(0.0, min(1.0, float(forecast_data['confidence'])))
        
        return Forecast(
            date=str(forecast_data['date']),
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            confidence=round(confidence, 2),
            reasoning=str(forecast_data['reasoning'])[:200]  # Limit reasoning length
        )


class EnsemblePredictor(IPredictor):
    """Ensemble predictor that combines multiple prediction models."""
    
    def __init__(self, predictors: List[IPredictor], weights: Optional[List[float]] = None):
        self.predictors = predictors
        self.weights = weights or [1.0] * len(predictors)
        
        if len(self.weights) != len(self.predictors):
            raise ValueError("Number of weights must match number of predictors")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    async def predict(
        self,
        analysis_results: List[AnalysisResult],
        **kwargs
    ) -> Forecast:
        """Generate ensemble forecast by combining multiple predictors."""
        try:
            symbol = analysis_results[0]['symbol'] if analysis_results else "UNKNOWN"
            LOGGER.info(f"Generating ensemble forecast for {symbol}")
            
            # Get predictions from all predictors
            predictions = []
            for i, predictor in enumerate(self.predictors):
                try:
                    prediction = await predictor.predict(analysis_results, **kwargs)
                    predictions.append(prediction)
                except Exception as e:
                    LOGGER.error(f"Predictor {i} failed: {e}")
                    # Continue with other predictors
            
            if not predictions:
                raise ValueError("All predictors failed")
            
            # Combine predictions using weighted average
            forecast = self._combine_predictions(predictions, symbol)
            
            LOGGER.info(f"Successfully generated ensemble forecast for {symbol}")
            return forecast
            
        except Exception as e:
            LOGGER.error(f"Ensemble prediction failed: {e}")
            return Forecast(
                date=datetime.now().strftime('%Y-%m-%d'),
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
                confidence=0.0,
                reasoning=f"Ensemble prediction failed: {str(e)}"
            )
    
    def _combine_predictions(self, predictions: List[Forecast], symbol: str) -> Forecast:
        """Combine multiple predictions using weighted average."""
        if len(predictions) == 1:
            return predictions[0]
        
        # Calculate weighted averages
        total_open = sum(p['open'] * w for p, w in zip(predictions, self.weights[:len(predictions)]))
        total_high = sum(p['high'] * w for p, w in zip(predictions, self.weights[:len(predictions)]))
        total_low = sum(p['low'] * w for p, w in zip(predictions, self.weights[:len(predictions)]))
        total_close = sum(p['close'] * w for p, w in zip(predictions, self.weights[:len(predictions)]))
        total_confidence = sum(p['confidence'] * w for p, w in zip(predictions, self.weights[:len(predictions)]))
        
        # Combine reasoning
        reasoning_parts = [f"Model {i+1}: {p['reasoning'][:50]}..." for i, p in enumerate(predictions)]
        combined_reasoning = f"Ensemble of {len(predictions)} models. " + " | ".join(reasoning_parts)
        
        # Use the most recent date
        latest_date = max(p['date'] for p in predictions)
        
        return Forecast(
            date=latest_date,
            open=round(total_open, 2),
            high=round(total_high, 2),
            low=round(total_low, 2),
            close=round(total_close, 2),
            confidence=round(total_confidence, 2),
            reasoning=combined_reasoning[:200]
        )


class PredictorFactory:
    """Factory for creating predictors."""
    
    @staticmethod
    def create_openai_predictor(
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.1
    ) -> OpenAIPredictor:
        """Create OpenAI predictor."""
        return OpenAIPredictor(api_key, model, temperature)
    
    @staticmethod
    def create_ensemble_predictor(
        predictors: List[IPredictor],
        weights: Optional[List[float]] = None
    ) -> EnsemblePredictor:
        """Create ensemble predictor."""
        return EnsemblePredictor(predictors, weights)
    
    @staticmethod
    def create_default_predictor() -> OpenAIPredictor:
        """Create default predictor using settings."""
        return OpenAIPredictor(
            api_key=SETTINGS.openai_api_key,
            model=SETTINGS.chat_model,
            temperature=SETTINGS.temperature
        ) 
