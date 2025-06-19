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
    """Advanced news and sentiment analyzer with comprehensive analysis."""
    
    def __init__(self):
        self.analysis_type = "sentiment_analysis"
        
        # Enhanced keyword sets for more accurate sentiment analysis
        self.positive_keywords = {
            'financial': ['profit', 'gain', 'earnings', 'revenue', 'growth', 'dividend', 'bonus', 'expansion', 'investment', 'acquisition', 'merger', 'ipo', 'fund', 'capital', 'cash', 'surplus'],
            'market': ['bull', 'rally', 'surge', 'rise', 'up', 'high', 'peak', 'boom', 'strong', 'robust', 'solid', 'outperform', 'beat', 'exceed', 'target'],
            'business': ['launch', 'new', 'innovation', 'technology', 'partnership', 'deal', 'contract', 'order', 'award', 'success', 'achievement', 'milestone', 'breakthrough'],
            'sentiment': ['positive', 'optimistic', 'confident', 'bullish', 'favorable', 'promising', 'bright', 'encouraging', 'upbeat', 'strong']
        }
        
        self.negative_keywords = {
            'financial': ['loss', 'decline', 'debt', 'deficit', 'cut', 'reduce', 'restructure', 'layoff', 'bankruptcy', 'default', 'penalty', 'fine'],
            'market': ['bear', 'crash', 'fall', 'drop', 'down', 'low', 'weak', 'plunge', 'slump', 'correction', 'volatile', 'underperform', 'miss'],
            'business': ['closure', 'shutdown', 'delay', 'postpone', 'cancel', 'suspend', 'dispute', 'lawsuit', 'investigation', 'scandal', 'breach'],
            'sentiment': ['negative', 'pessimistic', 'bearish', 'unfavorable', 'concerning', 'worried', 'cautious', 'weak', 'disappointing']
        }
        
        # Topic-based impact weights
        self.topic_weights = {
            'Financial Results': 1.5,
            'Funding Activities': 1.3,
            'Procurement and Sales': 1.2,
            'Capital Investment': 1.2,
            'Regulatory and Legal': 0.8,
            'New Offerings': 1.1,
            'Awards and Recognitions': 1.0,
            'Artificial Intelligence': 1.1,
            'Big Data/Analytics': 1.0
        }
    
    def analyze(self, data: StockData, news_data=None, **kwargs) -> AnalysisResult:
        """Perform comprehensive sentiment analysis on news data."""
        try:
            LOGGER.info(f"Starting enhanced sentiment analysis for {data['symbol']}")
            
            analysis_data = {
                'symbol': data['symbol'],
                'news_available': news_data is not None,
                'sentiment_score': 0.0,
                'sentiment_label': 'NEUTRAL',
                'news_count': 0,
                'analysis_summary': {},
                'detailed_analysis': {},
                'news_headlines': [],
                'topics_analysis': {},
                'temporal_analysis': {},
                'source_analysis': {},
                'confidence_factors': {}
            }
            
            if news_data and news_data.get('news'):
                news_items = news_data['news']
                analysis_data['news_count'] = len(news_items)
                
                # Perform comprehensive analysis
                sentiment_results = self._analyze_comprehensive_sentiment(news_items)
                temporal_analysis = self._analyze_temporal_patterns(news_items)
                topic_analysis = self._analyze_topics(news_items)
                source_analysis = self._analyze_sources(news_items)
                relevance_analysis = self._analyze_relevance(news_items, data['symbol'])
                
                # Compile comprehensive analysis
                analysis_data.update({
                    'sentiment_score': sentiment_results['overall_score'],
                    'sentiment_label': sentiment_results['label'],
                    'analysis_summary': {
                        'positive_signals': sentiment_results['positive_count'],
                        'negative_signals': sentiment_results['negative_count'],
                        'neutral_signals': sentiment_results['neutral_count'],  
                        'weighted_score': sentiment_results['weighted_score'],
                        'high_impact_news': sentiment_results['high_impact_count'],
                    },
                    'detailed_analysis': {
                        'category_scores': sentiment_results['category_scores'],
                        'keyword_matches': sentiment_results['keyword_matches'],
                        'impact_distribution': sentiment_results['impact_distribution']
                    },
                    'news_headlines': sentiment_results['headlines'][:10],  # Top 10 headlines for context
                    'topics_analysis': topic_analysis,
                    'temporal_analysis': temporal_analysis,
                    'source_analysis': source_analysis,
                    'relevance_analysis': relevance_analysis,
                    'confidence_factors': {
                        'news_volume': min(len(news_items) / 20.0, 1.0),  # Normalized to 20 news items
                        'source_diversity': len(source_analysis) / max(len(news_items), 1),
                        'topic_relevance': relevance_analysis.get('relevance_score', 0),
                        'temporal_freshness': temporal_analysis.get('freshness_score', 0)
                    }
                })
                
                # Calculate overall confidence
                confidence_factors = analysis_data['confidence_factors']
                confidence = (
                    confidence_factors['news_volume'] * 0.3 +
                    confidence_factors['source_diversity'] * 0.2 +
                    confidence_factors['topic_relevance'] * 0.3 + 
                    confidence_factors['temporal_freshness'] * 0.2
                )
            else:
                confidence = 0.1
            
            return AnalysisResult(
                symbol=data['symbol'],
                analysis_type=self.analysis_type,
                data=analysis_data,
                timestamp=datetime.now(),
                confidence=min(confidence, 1.0)
            )
            
        except Exception as e:
            LOGGER.error(f"Enhanced sentiment analysis failed for {data['symbol']}: {e}")
            return AnalysisResult(
                symbol=data['symbol'],
                analysis_type=self.analysis_type,
                data={'error': str(e)},
                timestamp=datetime.now(),
                confidence=0.0
            )
    
    def _analyze_comprehensive_sentiment(self, news_items: List[Dict]) -> Dict:
        """Perform comprehensive sentiment analysis on news items."""
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        category_scores = {category: 0 for category in self.positive_keywords.keys()}
        keyword_matches = {'positive': [], 'negative': []}
        impact_distribution = {'high': 0, 'medium': 0, 'low': 0}
        headlines = []
        high_impact_count = 0
        
        total_weighted_score = 0
        total_weight = 0
        
        for item in news_items[:20]:  # Analyze top 20 news items
            title = item.get('title', '').lower()
            summary = item.get('summary', '').lower()
            text = f"{title} {summary}"
            topics = item.get('topics', [])
            
            # Calculate topic weight
            topic_weight = 1.0
            for topic in topics:
                if topic in self.topic_weights:
                    topic_weight = max(topic_weight, self.topic_weights[topic])
            
            # Analyze sentiment for this item
            item_positive_score = 0
            item_negative_score = 0
            
            # Category-based analysis
            for category, keywords in self.positive_keywords.items():
                category_matches = sum(1 for keyword in keywords if keyword in text)
                item_positive_score += category_matches
                category_scores[category] += category_matches
                keyword_matches['positive'].extend([kw for kw in keywords if kw in text])
            
            for category, keywords in self.negative_keywords.items():
                category_matches = sum(1 for keyword in keywords if keyword in text)
                item_negative_score += category_matches
                keyword_matches['negative'].extend([kw for kw in keywords if kw in text])
            
            # Calculate item sentiment
            if item_positive_score > item_negative_score:
                positive_count += 1
                item_sentiment = (item_positive_score - item_negative_score) / max(item_positive_score + item_negative_score, 1)
            elif item_negative_score > item_positive_score:
                negative_count += 1
                item_sentiment = (item_positive_score - item_negative_score) / max(item_positive_score + item_negative_score, 1)
            else:
                neutral_count += 1
                item_sentiment = 0
            
            # Apply topic weighting
            weighted_sentiment = item_sentiment * topic_weight
            total_weighted_score += weighted_sentiment
            total_weight += topic_weight
            
            # Determine impact level
            impact_score = (item_positive_score + item_negative_score) * topic_weight
            if impact_score >= 5:
                impact_distribution['high'] += 1
                high_impact_count += 1
            elif impact_score >= 2:
                impact_distribution['medium'] += 1
            else:
                impact_distribution['low'] += 1
            
            # Store headline with sentiment for context
            headlines.append({
                'title': item.get('title', ''),
                'sentiment': item_sentiment,
                'impact': impact_score,
                'topics': topics,
                'pub_date': item.get('pub_date', ''),
                'source': item.get('source', '')
            })
        
        # Calculate overall scores
        overall_score = total_weighted_score / max(total_weight, 1) if total_weight > 0 else 0
        
        # Determine sentiment label
        if overall_score > 0.15:
            sentiment_label = 'POSITIVE'
        elif overall_score < -0.15:
            sentiment_label = 'NEGATIVE'
        else:
            sentiment_label = 'NEUTRAL'
        
        return {
            'overall_score': overall_score,
            'weighted_score': total_weighted_score,
            'label': sentiment_label,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'category_scores': category_scores,
            'keyword_matches': {
                'positive': list(set(keyword_matches['positive'])),
                'negative': list(set(keyword_matches['negative']))
            },
            'impact_distribution': impact_distribution,
            'headlines': sorted(headlines, key=lambda x: x['impact'], reverse=True),
            'high_impact_count': high_impact_count
        }
    
    def _analyze_temporal_patterns(self, news_items: List[Dict]) -> Dict:
        """Analyze temporal patterns in news."""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        time_buckets = {
            'last_24h': 0,
            'last_48h': 0,  
            'last_week': 0,
            'older': 0
        }
        
        for item in news_items:
            pub_date_str = item.get('pub_date', '')
            if pub_date_str:
                try:
                    # Parse ISO format date
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00')).replace(tzinfo=None)
                    time_diff = now - pub_date
                    
                    if time_diff <= timedelta(days=1):
                        time_buckets['last_24h'] += 1
                    elif time_diff <= timedelta(days=2):
                        time_buckets['last_48h'] += 1
                    elif time_diff <= timedelta(days=7):
                        time_buckets['last_week'] += 1
                    else:
                        time_buckets['older'] += 1
                except:
                    time_buckets['older'] += 1
        
        # Calculate freshness score (higher weight for recent news)
        total_news = sum(time_buckets.values())
        if total_news > 0:
            freshness_score = (
                time_buckets['last_24h'] * 1.0 +
                time_buckets['last_48h'] * 0.8 +
                time_buckets['last_week'] * 0.5 +
                time_buckets['older'] * 0.2
            ) / total_news
        else:
            freshness_score = 0
        
        return {
            'time_distribution': time_buckets,  
            'freshness_score': freshness_score,
            'total_analyzed': total_news
        }
    
    def _analyze_topics(self, news_items: List[Dict]) -> Dict:
        """Analyze news topics and their impact."""
        topic_counts = {}
        topic_sentiment = {}
        
        for item in news_items:
            topics = item.get('topics', [])
            title = item.get('title', '').lower()
            
            for topic in topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                # Simple sentiment for topic
                positive_words = sum(1 for word in self.positive_keywords.get('sentiment', []) if word in title)
                negative_words = sum(1 for word in self.negative_keywords.get('sentiment', []) if word in title)
                
                if topic not in topic_sentiment:
                    topic_sentiment[topic] = {'positive': 0, 'negative': 0}
                
                topic_sentiment[topic]['positive'] += positive_words
                topic_sentiment[topic]['negative'] += negative_words
        
        return {
            'topic_distribution': topic_counts,
            'topic_sentiment': topic_sentiment,
            'high_impact_topics': [topic for topic, count in topic_counts.items() if count >= 3]
        }
    
    def _analyze_sources(self, news_items: List[Dict]) -> Dict:
        """Analyze news sources and their credibility."""
        source_counts = {}
        source_credibility = {
            'The Economic Times': 0.9,
            'Business Standard': 0.9,
            'CNBC TV18': 0.85,
            'Moneycontrol': 0.85,
            'The Hindu BusinessLine': 0.8,
            'Financial Express': 0.8,
            'Zee Business': 0.75,
            'Inc42': 0.7
        }
        
        for item in news_items:
            source = item.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Calculate weighted credibility
        total_credibility = 0
        total_items = 0
        
        for source, count in source_counts.items():
            credibility = source_credibility.get(source, 0.6)  # Default credibility
            total_credibility += credibility * count
            total_items += count
        
        avg_credibility = total_credibility / max(total_items, 1)
        
        return {
            'source_distribution': source_counts,
            'average_credibility': avg_credibility,
            'credible_sources': len([s for s in source_counts.keys() if s in source_credibility])
        }
    
    def _analyze_relevance(self, news_items: List[Dict], symbol: str) -> Dict:
        """Analyze relevance of news to the specific stock."""
        company_name = symbol.replace('.NS', '').lower()
        
        # Common company name variations
        name_variations = [
            company_name,
            company_name.replace('_', ' '),
            company_name.replace('-', ' ')
        ]
        
        # Sector-specific keywords for BHEL
        if 'bhel' in company_name:
            sector_keywords = ['power', 'electricity', 'electrical', 'energy', 'transmission', 'grid', 'renewable', 'solar', 'wind', 'infrastructure', 'engineering']
        else:
            sector_keywords = ['industry', 'manufacturing', 'business', 'corporate']
        
        direct_mentions = 0
        indirect_mentions = 0
        sector_mentions = 0
        
        for item in news_items:
            title = item.get('title', '').lower()
            summary = item.get('summary', '').lower()
            text = f"{title} {summary}"
            
            # Check for direct company mentions
            for name_var in name_variations:
                if name_var in text:
                    direct_mentions += 1
                    break
            
            # Check for sector-related mentions  
            sector_count = sum(1 for keyword in sector_keywords if keyword in text)
            if sector_count > 0:
                sector_mentions += sector_count
            
            # Check for indirect business mentions
            if any(word in text for word in ['stock', 'market', 'share', 'equity', 'investment']):
                indirect_mentions += 1
        
        # Calculate relevance score
        total_items = len(news_items)
        relevance_score = 0
        if total_items > 0:
            relevance_score = (
                (direct_mentions * 1.0) +
                (sector_mentions * 0.5) + 
                (indirect_mentions * 0.2)
            ) / total_items
        
        return {
            'direct_mentions': direct_mentions,
            'sector_mentions': sector_mentions,
            'indirect_mentions': indirect_mentions,
            'relevance_score': min(relevance_score, 1.0),
            'sector_keywords_found': sector_keywords
        }


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
