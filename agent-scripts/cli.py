"""
Buck_V1.cli
────────────────────────
Command-line interface for the Buck.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from .buck import BuckFactory
from .config import LOGGER


def setup_cli() -> argparse.ArgumentParser:
    """Setup command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Stock Analysis and Prediction Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single stock
  python -m market_forecaster.cli analyze BHEL.NS --start 2024-01-01 --end 2024-01-10
  
  # Batch analyze multiple stocks
  python -m market_forecaster.cli batch BHEL.NS RELIANCE.NS TCS.NS --start 2024-01-01 --end 2024-01-10
  
  # Use specific model and interval
  python -m market_forecaster.cli analyze BHEL.NS --start 2024-01-01 --end 2024-01-10 --model gpt-4o --interval 30m
        """)
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single stock')
    analyze_parser.add_argument('symbol', help='Stock symbol (e.g., BHEL.NS)')
    analyze_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    analyze_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    analyze_parser.add_argument('--interval', default='1h', 
                              choices=['5m', '30m', '1h', '4h', '6h', '8h', '12h', '1d'],
                              help='Data interval (default: 1h)')
    analyze_parser.add_argument('--model', default='gpt-4o', help='OpenAI model to use')
    analyze_parser.add_argument('--no-save', action='store_true', help='Don\'t save results to files')
    analyze_parser.add_argument('--output', help='Custom output directory')
    
    # Batch analysis command
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple stocks')
    batch_parser.add_argument('symbols', nargs='+', help='Stock symbols (e.g., BHEL.NS RELIANCE.NS)')
    batch_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    batch_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    batch_parser.add_argument('--interval', default='1h',
                             choices=['5m', '30m', '1h', '4h', '6h', '8h', '12h', '1d'],
                             help='Data interval (default: 1h)')
    batch_parser.add_argument('--model', default='gpt-4o', help='OpenAI model to use')
    batch_parser.add_argument('--concurrent', type=int, default=3, help='Max concurrent analyses')
    batch_parser.add_argument('--output', help='Custom output directory')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a demo analysis')
    demo_parser.add_argument('--symbol', default='BHEL.NS', help='Symbol for demo (default: BHEL.NS)')
    
    # Configuration
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--indian-api-key', help='Indian API key for news data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    return parser


async def run_single_analysis(args) -> None:
    """Run single stock analysis."""
    try:
        print(f"🔍 Analyzing {args.symbol} from {args.start} to {args.end}...")
        
        # Create agent
        if args.api_key:
            agent = BuckFactory.create_production_agent(
                openai_api_key=args.api_key,
                indian_api_key=args.indian_api_key or "",
                model=args.model
            )
        else:
            agent = BuckFactory.create_default_agent()
        
        # Run analysis
        async with agent:
            results = await agent.analyze_and_predict(
                symbol=args.symbol,
                start_date=args.start,
                end_date=args.end,
                interval=args.interval,
                save_results=not args.no_save
            )
        
        # Display results
        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
            return
        
        forecast = results['forecast']
        metadata = results['metadata']
        
        print(f"\n📈 Analysis Results for {args.symbol}")
        print("=" * 50)
        print(f"Data Period: {args.start} to {args.end} ({args.interval})")
        print(f"Data Points: {results['data_info']['data_points']}")
        print(f"News Available: {'Yes' if results['data_info']['news_available'] else 'No'}")
        print(f"Model Used: {metadata['model_used']}")
        
        print(f"\n🔮 Next-Day Forecast ({forecast['date']})")
        print("-" * 30)
        print(f"Open:  ₹{forecast['open']:.2f}")
        print(f"High:  ₹{forecast['high']:.2f}")
        print(f"Low:   ₹{forecast['low']:.2f}")
        print(f"Close: ₹{forecast['close']:.2f}")
        print(f"Confidence: {forecast['confidence']:.2f}")
        print(f"Reasoning: {forecast['reasoning']}")
        
        print(f"\n📊 Confidence Scores")
        print("-" * 20)
        print(f"Analysis: {metadata['analysis_confidence']:.2f}")
        print(f"Prediction: {metadata['prediction_confidence']:.2f}")
        
        if not args.no_save:
            print(f"\n💾 Results saved to output directory")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        LOGGER.error(f"CLI single analysis error: {e}")


async def run_batch_analysis(args) -> None:
    """Run batch analysis."""
    try:
        print(f"🔍 Batch analyzing {len(args.symbols)} stocks from {args.start} to {args.end}...")
        
        # Create agent
        if args.api_key:
            agent = BuckFactory.create_production_agent(
                openai_api_key=args.api_key,
                indian_api_key=args.indian_api_key or "",
                model=args.model
            )
        else:
            agent = BuckFactory.create_default_agent()
        
        # Run batch analysis
        async with agent:
            results = await agent.batch_analyze(
                symbols=args.symbols,
                start_date=args.start,
                end_date=args.end,
                interval=args.interval,
                max_concurrent=args.concurrent
            )
        
        # Display results
        summary = results['summary']
        print(f"\n📊 Batch Analysis Summary")
        print("=" * 40)
        print(f"Total Symbols: {len(args.symbols)}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Average Confidence: {summary['avg_confidence']:.2f}")
        
        print(f"\n📈 Individual Results")
        print("-" * 40)
        
        for symbol in args.symbols:
            if symbol in results['results']:
                result = results['results'][symbol]
                if 'error' in result:
                    print(f"{symbol}: ❌ Failed - {result['error']}")
                else:
                    forecast = result['forecast']
                    confidence = result['metadata']['prediction_confidence']
                    print(f"{symbol}: ₹{forecast['close']:.2f} (conf: {confidence:.2f})")
            else:
                print(f"{symbol}: ❌ No result")
        
        # Save batch results
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path('output')
            
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_file = output_dir / f"batch_analysis_{timestamp}.json"
        
        with batch_file.open('w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Batch results saved to {batch_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        LOGGER.error(f"CLI batch analysis error: {e}")


async def run_demo(args) -> None:
    """Run demo analysis."""
    try:
        print(f"🚀 Running demo analysis for {args.symbol}...")
        
        # Use recent dates for demo
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        # Create demo args
        demo_args = argparse.Namespace(
            symbol=args.symbol,
            start=start_date,
            end=end_date,
            interval='1h',
            model='gpt-4o',
            no_save=False,
            output=None,
            api_key=args.api_key,
            indian_api_key=args.indian_api_key
        )
        
        await run_single_analysis(demo_args)
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        LOGGER.error(f"CLI demo error: {e}")


async def main() -> None:
    """Main CLI entry point."""
    parser = setup_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    if args.verbose:
        import logging
        logging.getLogger("market_forecaster").setLevel(logging.DEBUG)
    
    # Validate API key
    if not args.api_key:
        try:
            from .config import SETTINGS
            if not SETTINGS.openai_api_key:
                print("❌ Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
                return
        except Exception:
            print("❌ Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
            return
    
    # Route to appropriate command
    try:
        if args.command == 'analyze':
            await run_single_analysis(args)
        elif args.command == 'batch':
            await run_batch_analysis(args)
        elif args.command == 'demo':
            await run_demo(args)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        LOGGER.error(f"CLI main error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 
