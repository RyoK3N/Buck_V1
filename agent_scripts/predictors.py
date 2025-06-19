"""
Buck_V1.predictors
──────────────────────────────
AI-powered stock prediction implementations.
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
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
        self._json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
    
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
            
            # Save input data to inputs folder
            await self._save_input_data(symbol, system_prompt, user_prompt, analysis_results)
            
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
    
    async def _save_input_data(
        self, 
        symbol: str, 
        system_prompt: str, 
        user_prompt: str, 
        analysis_results: List[AnalysisResult]
    ) -> None:
        """Save the input data sent to OpenAI for inspection."""
        try:
            # Create inputs directory
            inputs_dir = Path('inputs')
            inputs_dir.mkdir(exist_ok=True)
            
            # Create timestamp for unique filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Prepare input data structure
            input_data = {
                'metadata': {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'model': self.model,
                    'temperature': self.temperature,
                    'analysis_results_count': len(analysis_results)
                },
                'openai_input': {
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'model_config': {
                        'model': self.model,
                        'temperature': self.temperature,
                        'response_format': {"type": "json_object"}
                    }
                },
                'analysis_results': analysis_results
            }
            
            # Save complete input data
            input_file = inputs_dir / f"{symbol}_{timestamp}_openai_input.json"
            with input_file.open('w', encoding='utf-8') as f:
                json.dump(input_data, f, indent=2, default=str, ensure_ascii=False)
            
            # Save prompts separately for easy viewing
            prompts_file = inputs_dir / f"{symbol}_{timestamp}_prompts.txt"
            with prompts_file.open('w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"OPENAI INPUT DATA FOR {symbol}\n")
                f.write(f"Generated at: {datetime.now().isoformat()}\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("SYSTEM PROMPT:\n")
                f.write("-" * 40 + "\n")
                f.write(system_prompt)
                f.write("\n\n")
                
                f.write("USER PROMPT (ANALYSIS DATA):\n")
                f.write("-" * 40 + "\n")
                f.write(user_prompt)
                f.write("\n\n")
                
                f.write("RAW ANALYSIS RESULTS:\n")
                f.write("-" * 40 + "\n")
                f.write(json.dumps(analysis_results, indent=2, default=str))
            
            LOGGER.info(f"OpenAI input data saved to inputs/{symbol}_{timestamp}_openai_input.json")
            LOGGER.info(f"Human-readable prompts saved to inputs/{symbol}_{timestamp}_prompts.txt")
            
        except Exception as e:
            LOGGER.error(f"Failed to save OpenAI input data: {e}")
            # Don't fail the prediction if saving inputs fails
    
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
            LOGGER.debug(f"Raw OpenAI response length: {len(raw_content)} chars")
            
            # Pre-process content to fix common OpenAI JSON formatting issues
            processed_content = self._preprocess_openai_response(raw_content)
            
            try:
                # First attempt: direct parsing on processed content
                return json.loads(processed_content)
            except json.JSONDecodeError as e:
                LOGGER.warning(f"Direct JSON parse failed: {e}")
                
                # Second attempt: clean and extract JSON
                cleaned_content = processed_content.strip()
                
                # Find JSON boundaries
                start_idx = cleaned_content.find('{')
                end_idx = cleaned_content.rfind('}')
                
                if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                    raise ValueError(f"No valid JSON object found in response")
                
                json_str = cleaned_content[start_idx:end_idx+1]
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Third attempt: manual reconstruction for OpenAI forecast format
                    LOGGER.warning("Attempting manual JSON reconstruction")
                    
                    try:
                        # Extract key fields using regex patterns
                        import re
                        
                        # Extract each field with better patterns
                        date_match = re.search(r'"date":\s*"([^"]+)"', json_str)
                        open_match = re.search(r'"open":\s*([0-9.]+)', json_str)
                        high_match = re.search(r'"high":\s*([0-9.]+)', json_str)
                        low_match = re.search(r'"low":\s*([0-9.]+)', json_str)
                        close_match = re.search(r'"close":\s*([0-9.]+)', json_str)
                        confidence_match = re.search(r'"confidence":\s*([0-9.]+)', json_str)
                        
                        # Extract reasoning - handle multiline content
                        reasoning_match = re.search(r'"reasoning":\s*"(.*?)"(?:\s*[,}])', json_str, re.DOTALL)
                        if not reasoning_match:
                            # Try alternative pattern if reasoning is at the end
                            reasoning_match = re.search(r'"reasoning":\s*"(.*?)$', json_str, re.DOTALL)
                        
                        if not all([date_match, open_match, high_match, low_match, close_match, confidence_match]):
                            raise ValueError("Could not extract all required fields from response")
                        
                        # Clean up reasoning text
                        reasoning_text = reasoning_match.group(1) if reasoning_match else "Analysis not available"
                        reasoning_text = reasoning_text.replace('\n', ' ').replace('\\n', ' ').strip()
                        # Remove any trailing incomplete sentences or quotes
                        if reasoning_text.endswith(('the', 'and', 'but', 'with', 'for', 'of', 'in', 'to')):
                            words = reasoning_text.split()
                            reasoning_text = ' '.join(words[:-1]) + '.'
                        
                        return {
                            "date": date_match.group(1),
                            "open": float(open_match.group(1)),
                            "high": float(high_match.group(1)),
                            "low": float(low_match.group(1)),
                            "close": float(close_match.group(1)),
                            "confidence": float(confidence_match.group(1)),
                            "reasoning": reasoning_text
                        }
                        
                    except Exception as manual_error:
                        LOGGER.error(f"Manual JSON reconstruction failed: {manual_error}")
                        raise ValueError(f"Could not parse OpenAI response. Raw content preview: {raw_content[:300]}...")
            
        except Exception as e:
            LOGGER.error(f"OpenAI sync call failed: {e}")
            raise
    
    def _preprocess_openai_response(self, raw_content: str) -> str:
        """Preprocess OpenAI response to fix common JSON formatting issues."""
        import re
        
        # Fix the main issue: trailing backslashes in reasoning field
        # Remove trailing backslashes that break JSON parsing
        content = raw_content.strip()
        
        # Pattern to find and fix unterminated strings with trailing backslashes
        # This specifically handles the case where reasoning ends with "\ instead of "
        content = re.sub(r'(\"reasoning\":\s*\".*?)\\(\s*)$', r'\1"\2}', content, flags=re.DOTALL)
        
        # Also fix cases where the backslash is followed by whitespace and then EOF
        content = re.sub(r'(\"reasoning\":\s*\".*?)\\(\s*[\r\n]*\s*)$', r'\1"\2}', content, flags=re.DOTALL)
        
        # Make sure we have a proper closing brace
        if not content.rstrip().endswith('}'):
            content = content.rstrip() + '}'
        
        return content
    
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
- "reasoning": Detailed analysis reasoning (string, provide comprehensive rationale explaining your prediction, including specific technical indicators, sentiment factors, and risk considerations - aim for 300-500 words)

Guidelines:
1. Base predictions on technical indicators (RSI, MACD, moving averages, support/resistance)
2. Factor in volume patterns (OBV), candlestick signals, and momentum indicators
3. Consider sentiment analysis results, news impact, and market psychology
4. Ensure high >= max(open, close) and low <= min(open, close)
5. Explain your confidence level based on data quality and signal clarity
6. Be conservative with price movements (typically 0.5-3% daily)
7. In reasoning, specifically mention which indicators support your prediction and any conflicting signals

Provide detailed reasoning that explains the logic behind your price targets and confidence score.

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
        """Format analysis results into a comprehensive prompt with enhanced detail."""
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
        tools_results = technical_data.get('tools_results', {})
        
        # Format technical indicators with more detail
        technical_section = f"""### Technical Analysis for {symbol}
Overall Signal: {summary.get('overall_signal', 'N/A')} (Strength: {self._safe_format_float(summary.get('signal_strength', 0))})

Key Indicators:
- RSI: {self._safe_format_float(key_metrics.get('rsi', 'N/A'))} ({key_metrics.get('rsi_condition', 'N/A')})
- MACD: {self._safe_format_float(key_metrics.get('macd', 'N/A'), 4)} | Signal: {self._safe_format_float(key_metrics.get('macd_signal', 'N/A'), 4)}
- MACD Histogram: {self._safe_format_float(key_metrics.get('macd_histogram', 'N/A'), 4)}
- Moving Averages: Short MA: {self._safe_format_float(key_metrics.get('short_ma', 'N/A'))} | Long MA: {self._safe_format_float(key_metrics.get('long_ma', 'N/A'))}
- Support: {key_metrics.get('nearest_support', 'N/A')} | Resistance: {key_metrics.get('nearest_resistance', 'N/A')}
- Volume Analysis (OBV): {self._safe_format_float(tools_results.get('obv', {}).get('obv', 'N/A'))} | Trend: {tools_results.get('obv', {}).get('obv_trend', 'N/A')}
- Candlestick Patterns: {key_metrics.get('pattern_count', 0)} detected
  - Bullish Score: {self._safe_format_float(key_metrics.get('bullish_score', 0), 1)}
  - Bearish Score: {self._safe_format_float(key_metrics.get('bearish_score', 0), 1)}

Signal Breakdown: {summary.get('signals_breakdown', {})}
Tools Processed: {summary.get('tools_processed', 0)}/{summary.get('tools_processed', 0) + summary.get('tools_failed', 0)}"""

        # Enhanced sentiment analysis section with comprehensive details
        sentiment_summary = sentiment_data.get('analysis_summary', {})
        detailed_analysis = sentiment_data.get('detailed_analysis', {})
        topics_analysis = sentiment_data.get('topics_analysis', {})
        temporal_analysis = sentiment_data.get('temporal_analysis', {})
        source_analysis = sentiment_data.get('source_analysis', {})
        relevance_analysis = sentiment_data.get('relevance_analysis', {})
        confidence_factors = sentiment_data.get('confidence_factors', {})
        
        sentiment_section = f"""### Enhanced Sentiment Analysis
News Available: {sentiment_data.get('news_available', False)}
News Count: {sentiment_data.get('news_count', 0)}
Overall Sentiment: {sentiment_data.get('sentiment_label', 'NEUTRAL')} (Score: {self._safe_format_float(sentiment_data.get('sentiment_score', 0), 3)})

Signal Distribution:
- Positive Signals: {sentiment_summary.get('positive_signals', 0)}
- Negative Signals: {sentiment_summary.get('negative_signals', 0)}
- Neutral Signals: {sentiment_summary.get('neutral_signals', 0)}
- High Impact News: {sentiment_summary.get('high_impact_news', 0)}
- Weighted Sentiment Score: {self._safe_format_float(sentiment_summary.get('weighted_score', 0), 3)}

News Impact Analysis:
- High Impact: {detailed_analysis.get('impact_distribution', {}).get('high', 0)} items
- Medium Impact: {detailed_analysis.get('impact_distribution', {}).get('medium', 0)} items
- Low Impact: {detailed_analysis.get('impact_distribution', {}).get('low', 0)} items

Topic Distribution: {topics_analysis.get('topic_distribution', {})}
High Impact Topics: {topics_analysis.get('high_impact_topics', [])}

Temporal Pattern:
- Last 24h: {temporal_analysis.get('time_distribution', {}).get('last_24h', 0)} items
- Last 48h: {temporal_analysis.get('time_distribution', {}).get('last_48h', 0)} items
- Last Week: {temporal_analysis.get('time_distribution', {}).get('last_week', 0)} items
- News Freshness Score: {self._safe_format_float(temporal_analysis.get('freshness_score', 0), 2)}

Source Analysis:
- Source Diversity: {len(source_analysis.get('source_distribution', {}))} unique sources
- Average Credibility: {self._safe_format_float(source_analysis.get('average_credibility', 0), 2)}
- Credible Sources: {source_analysis.get('credible_sources', 0)}

Relevance Analysis:
- Direct Company Mentions: {relevance_analysis.get('direct_mentions', 0)}
- Sector Mentions: {relevance_analysis.get('sector_mentions', 0)}
- Relevance Score: {self._safe_format_float(relevance_analysis.get('relevance_score', 0), 2)}

Confidence Factors:
- News Volume: {self._safe_format_float(confidence_factors.get('news_volume', 0), 2)}
- Source Diversity: {self._safe_format_float(confidence_factors.get('source_diversity', 0), 2)}
- Topic Relevance: {self._safe_format_float(confidence_factors.get('topic_relevance', 0), 2)}
- Temporal Freshness: {self._safe_format_float(confidence_factors.get('temporal_freshness', 0), 2)}"""

        # Add recent news headlines for context
        headlines_section = ""
        news_headlines = sentiment_data.get('news_headlines', [])
        if news_headlines:
            headlines_section = f"""
Recent High-Impact Headlines:"""
            for i, headline in enumerate(news_headlines[:5], 1):
                sentiment_label = "📈" if headline.get('sentiment', 0) > 0 else "📉" if headline.get('sentiment', 0) < 0 else "➡️"
                headlines_section += f"""
{i}. {sentiment_label} "{headline.get('title', '')}"
   Impact: {self._safe_format_float(headline.get('impact', 0), 1)} | Source: {headline.get('source', 'Unknown')}
   Topics: {headline.get('topics', [])} | Date: {headline.get('pub_date', 'Unknown')[:10]}"""

        # Enhanced market data context
        data_info = technical_data.get('data_info', {})
        market_context_section = f"""### Market Data Context
Analysis Period: {data_info.get('start_date', 'N/A')} to {data_info.get('end_date', 'N/A')}
Data Interval: {data_info.get('interval', 'N/A')}
Data Points Available: {data_info.get('data_points', 'N/A')}
Data Quality: {'Sufficient' if data_info.get('data_points', 0) >= 20 else 'Limited' if data_info.get('data_points', 0) >= 5 else 'Minimal'}"""

        # Try to extract current price from tools results
        for tool_name, result in tools_results.items():
            if 'current_price' in result:
                current_price = result['current_price']
                break
        
        if current_price:
            price_section = f"Current Market Price: ₹{self._safe_format_float(current_price, 2)}"
        else:
            price_section = "Current Market Price: Not available"

        # Add prediction guidance based on data quality
        analysis_quality = self._assess_analysis_quality(technical_data, sentiment_data)
        
        guidance_section = f"""
### Analysis Quality Assessment
Technical Analysis Quality: {analysis_quality['technical_quality']}
Sentiment Analysis Quality: {analysis_quality['sentiment_quality']}
Overall Data Confidence: {analysis_quality['overall_confidence']}
Key Limitations: {analysis_quality['limitations']}
Prediction Confidence Guidance: {analysis_quality['confidence_guidance']}"""

        return f"""{market_context_section}

{price_section}

{technical_section}

{sentiment_section}{headlines_section}

{guidance_section}

Based on this comprehensive analysis combining technical indicators, enhanced sentiment analysis with temporal patterns, source credibility, relevance scoring, and market context, please provide your next-day forecast. Pay special attention to:
1. Technical signal consistency and strength
2. News sentiment momentum and impact levels  
3. Temporal patterns in news flow
4. Source credibility and relevance scores
5. Data quality limitations and confidence factors

Your forecast should reflect the nuanced interplay between these factors."""
    
    def _assess_analysis_quality(self, technical_data: Dict, sentiment_data: Dict) -> Dict:
        """Assess the quality and reliability of the analysis data."""
        data_info = technical_data.get('data_info', {})
        tools_results = technical_data.get('tools_results', {})
        confidence_factors = sentiment_data.get('confidence_factors', {})
        
        # Assess technical analysis quality
        data_points = data_info.get('data_points', 0)
        nan_indicators = sum(1 for tool, result in tools_results.items() 
                           if any(str(value) == 'nan' for value in str(result).split()))
        
        if data_points >= 50 and nan_indicators <= 1:
            technical_quality = "High"
        elif data_points >= 20 and nan_indicators <= 3:
            technical_quality = "Medium"
        elif data_points >= 5:
            technical_quality = "Low"
        else:
            technical_quality = "Very Low"
        
        # Assess sentiment analysis quality
        news_count = sentiment_data.get('news_count', 0)
        relevance_score = confidence_factors.get('topic_relevance', 0)
        source_diversity = confidence_factors.get('source_diversity', 0)
        
        if news_count >= 15 and relevance_score >= 0.3 and source_diversity >= 0.5:
            sentiment_quality = "High"
        elif news_count >= 10 and relevance_score >= 0.2:
            sentiment_quality = "Medium"
        elif news_count >= 5:
            sentiment_quality = "Low"
        else:
            sentiment_quality = "Very Low"
        
        # Overall assessment
        quality_scores = {"High": 4, "Medium": 3, "Low": 2, "Very Low": 1}
        avg_quality = (quality_scores[technical_quality] + quality_scores[sentiment_quality]) / 2
        
        if avg_quality >= 3.5:
            overall_confidence = "High"
            confidence_guidance = "Strong confidence in predictions - comprehensive data available"
        elif avg_quality >= 2.5:
            overall_confidence = "Medium" 
            confidence_guidance = "Moderate confidence - some data limitations present"
        elif avg_quality >= 1.5:
            overall_confidence = "Low"
            confidence_guidance = "Low confidence - significant data constraints"
        else:
            overall_confidence = "Very Low"
            confidence_guidance = "Very low confidence - insufficient data for reliable predictions"
        
        # Identify limitations
        limitations = []
        if data_points < 20:
            limitations.append("Limited historical data")
        if nan_indicators > 2:
            limitations.append("Missing technical indicators")
        if news_count < 10:
            limitations.append("Limited news coverage")
        if relevance_score < 0.3:
            limitations.append("Low news relevance")
        if not limitations:
            limitations.append("None identified")
        
        return {
            'technical_quality': technical_quality,
            'sentiment_quality': sentiment_quality,
            'overall_confidence': overall_confidence,
            'confidence_guidance': confidence_guidance,
            'limitations': ', '.join(limitations)
        }
    
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
            reasoning=str(forecast_data['reasoning'])[:800]  # Increased from 200 to 800 characters for complete reasoning
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
            reasoning=combined_reasoning[:800]  # Increased from 200 to 800 characters for complete reasoning
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
