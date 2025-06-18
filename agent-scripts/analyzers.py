"""
Buck_V1.analyzers
──────────────────────────────
Stock analyzer implementations.
"""

from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

from .interfaces import IAnalyzer, AnalysisResult, StockData, ITool
from .tools import ToolFactory
from .config import LOGGER


class TechnicalAnalyzer(IAnalyzer):
    """Technical analysis coordinator using multiple tools."""
    
    def __init__(self, tools: Optional[Dict[str, ITool]] = None):
        self.tools = tools or ToolFactory.create_all_tools()
        self.analysis_type = "technical_analysis"
    
    def analyze(self, data: StockData, **kwargs) -> AnalysisResult:
        """Perform comprehensive technical analysis."""
        try:
            LOGGER.info(f"Starting technical analysis for {data['symbol']}")
            
            analysis_data = {
                'symbol': data['symbol'],
                'data_info': {
                    'interval': data['interval'],
                    'start_date': data['start_date'],
                    'end_date': data['end_date'],
                    'data_points': len(data['data'])
                },
                'tools_results': {},
                'summary': {}
            }
            
            # Run all tools
            for tool_name, tool in self.tools.items():
                try:
                    result = tool.execute(data['data'], **kwargs)
                    analysis_data['tools_results'][tool_name] = result
                except Exception as e:
                    LOGGER.error(f"Error running tool {tool_name}: {e}")
                    analysis_data['tools_results'][tool_name] = {'error': str(e)}
            
            # Generate summary
            analysis_data['summary'] = self._generate_summary(analysis_data['tools_results'])
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(analysis_data['tools_results'])
            
            return AnalysisResult(
                symbol=data['symbol'],
                analysis_type=self.analysis_type,
                data=analysis_data,
                timestamp=datetime.now(),
                confidence=confidence
            )
            
        except Exception as e:
            LOGGER.error(f"Technical analysis failed for {data['symbol']}: {e}")
            return AnalysisResult(
                symbol=data['symbol'],
                analysis_type=self.analysis_type,
                data={'error': str(e)},
                timestamp=datetime.now(),
                confidence=0.0
            )
    
    def _generate_summary(self, tools_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary from tool results."""
        signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_strength = 0
        valid_tools = 0
        
        key_metrics = {}
        
        for tool_name, result in tools_results.items():
            if 'error' in result:
                continue
                
            valid_tools += 1
            
            # Collect signals
            if 'signal' in result:
                signal = result['signal']
                strength = result.get('strength', 0.5)
                signals[signal] += strength
                total_strength += strength
            
            # Collect key metrics
            if tool_name == 'rsi':
                key_metrics['rsi'] = result.get('rsi')
                key_metrics['rsi_condition'] = result.get('condition')
            elif tool_name == 'macd':
                key_metrics['macd'] = result.get('macd')
                key_metrics['macd_signal'] = result.get('signal')
                key_metrics['macd_histogram'] = result.get('histogram')
            elif tool_name == 'moving_average':
                key_metrics['short_ma'] = result.get('short_ma')
                key_metrics['long_ma'] = result.get('long_ma')
            elif tool_name == 'support_resistance':
                key_metrics['nearest_support'] = result.get('nearest_support')
                key_metrics['nearest_resistance'] = result.get('nearest_resistance')
            elif tool_name == 'candlestick_patterns':
                key_metrics['pattern_count'] = result.get('pattern_count', 0)
                key_metrics['bullish_score'] = result.get('bullish_score', 0)
                key_metrics['bearish_score'] = result.get('bearish_score', 0)
        
        # Determine overall signal
        max_signal = max(signals, key=signals.get)
        signal_strength = float(signals[max_signal] / (total_strength or 1))
        
        return {
            'overall_signal': max_signal,
            'signal_strength': signal_strength,
            'signals_breakdown': {k: float(v) for k, v in signals.items()},
            'key_metrics': key_metrics,
            'tools_processed': valid_tools,
            'tools_failed': len(tools_results) - valid_tools
        }
    
    def _calculate_confidence(self, tools_results: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        successful_tools = sum(1 for result in tools_results.values() if 'error' not in result)
        total_tools = len(tools_results)
        
        if total_tools == 0:
            return 0.0
        
        # Base confidence on successful tool execution
        base_confidence = successful_tools / total_tools
        
        # Adjust based on signal consistency
        signals = []
        for result in tools_results.values():
            if 'error' not in result and 'signal' in result:
                signals.append(result['signal'])
        
        if signals:
            # Calculate signal consistency
            signal_counts = {}
            for signal in signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            max_count = max(signal_counts.values())
            consistency = max_count / len(signals)
            
            # Combine base confidence with consistency
            final_confidence = (base_confidence * 0.6) + (consistency * 0.4)
        else:
            final_confidence = base_confidence * 0.5  # Lower confidence if no signals
        
        return min(final_confidence, 1.0)


class SentimentAnalyzer(IAnalyzer):
    """News and sentiment analyzer."""
    
    def __init__(self):
        self.analysis_type = "sentiment_analysis"
    
    def analyze(self, data: StockData, news_data=None, **kwargs) -> AnalysisResult:
        """Perform sentiment analysis on news data."""
        try:
            LOGGER.info(f"Starting sentiment analysis for {data['symbol']}")
            
            analysis_data = {
                'symbol': data['symbol'],
                'news_available': news_data is not None,
                'sentiment_score': 0.0,
                'sentiment_label': 'NEUTRAL',
                'news_count': 0,
                'keywords': []
            }
            
            if news_data and news_data.get('news'):
                news_items = news_data['news']
                analysis_data['news_count'] = len(news_items)
                
                # Simple keyword-based sentiment analysis
                positive_keywords = ['growth', 'profit', 'gain', 'rise', 'bull', 'positive', 'strong', 'increase', 'up']
                negative_keywords = ['loss', 'decline', 'fall', 'bear', 'negative', 'weak', 'decrease', 'down', 'crash']
                
                positive_score = 0
                negative_score = 0
                
                for item in news_items[:10]:  # Analyze top 10 news items
                    title = item.get('title', '').lower()
                    summary = item.get('summary', '').lower()
                    text = f"{title} {summary}"
                    
                    for keyword in positive_keywords:
                        if keyword in text:
                            positive_score += 1
                    
                    for keyword in negative_keywords:
                        if keyword in text:
                            negative_score += 1
                
                # Calculate sentiment score (-1 to 1)
                total_score = positive_score + negative_score
                if total_score > 0:
                    sentiment_score = (positive_score - negative_score) / total_score
                else:
                    sentiment_score = 0.0
                
                # Determine sentiment label
                if sentiment_score > 0.2:
                    sentiment_label = 'POSITIVE'
                elif sentiment_score < -0.2:
                    sentiment_label = 'NEGATIVE'
                else:
                    sentiment_label = 'NEUTRAL'
                
                analysis_data.update({
                    'sentiment_score': sentiment_score,
                    'sentiment_label': sentiment_label,
                    'positive_mentions': positive_score,
                    'negative_mentions': negative_score
                })
            
            confidence = 0.8 if news_data else 0.1
            
            return AnalysisResult(
                symbol=data['symbol'],
                analysis_type=self.analysis_type,
                data=analysis_data,
                timestamp=datetime.now(),
                confidence=confidence
            )
            
        except Exception as e:
            LOGGER.error(f"Sentiment analysis failed for {data['symbol']}: {e}")
            return AnalysisResult(
                symbol=data['symbol'],
                analysis_type=self.analysis_type,
                data={'error': str(e)},
                timestamp=datetime.now(),
                confidence=0.0
            )


class CompositeAnalyzer(IAnalyzer):
    """Composite analyzer that combines multiple analysis types."""
    
    def __init__(self, analyzers: Optional[List[IAnalyzer]] = None):
        self.analyzers = analyzers or [
            TechnicalAnalyzer(),
            SentimentAnalyzer()
        ]
        self.analysis_type = "composite_analysis"
    
    def analyze(self, data: StockData, **kwargs) -> AnalysisResult:
        """Perform composite analysis using all analyzers."""
        try:
            LOGGER.info(f"Starting composite analysis for {data['symbol']}")
            
            analysis_results = {}
            total_confidence = 0
            
            for analyzer in self.analyzers:
                try:
                    result = analyzer.analyze(data, **kwargs)
                    analysis_results[analyzer.analysis_type] = result
                    total_confidence += result['confidence']
                except Exception as e:
                    LOGGER.error(f"Analyzer {analyzer.analysis_type} failed: {e}")
                    analysis_results[analyzer.analysis_type] = {
                        'error': str(e),
                        'confidence': 0.0
                    }
            
            # Calculate composite confidence
            avg_confidence = total_confidence / len(self.analyzers) if self.analyzers else 0.0
            
            composite_data = {
                'symbol': data['symbol'],
                'analysis_results': analysis_results,
                'composite_confidence': avg_confidence,
                'analyzers_count': len(self.analyzers)
            }
            
            return AnalysisResult(
                symbol=data['symbol'],
                analysis_type=self.analysis_type,
                data=composite_data,
                timestamp=datetime.now(),
                confidence=avg_confidence
            )
            
        except Exception as e:
            LOGGER.error(f"Composite analysis failed for {data['symbol']}: {e}")
            return AnalysisResult(
                symbol=data['symbol'],
                analysis_type=self.analysis_type,
                data={'error': str(e)},
                timestamp=datetime.now(),
                confidence=0.0
            )


class AnalyzerFactory:
    """Factory for creating analyzers."""
    
    @staticmethod
    def create_technical_analyzer(tools: Optional[Dict[str, ITool]] = None) -> TechnicalAnalyzer:
        """Create technical analyzer."""
        return TechnicalAnalyzer(tools)
    
    @staticmethod
    def create_sentiment_analyzer() -> SentimentAnalyzer:
        """Create sentiment analyzer."""
        return SentimentAnalyzer()
    
    @staticmethod
    def create_composite_analyzer(analyzers: Optional[List[IAnalyzer]] = None) -> CompositeAnalyzer:
        """Create composite analyzer."""
        return CompositeAnalyzer(analyzers) 
