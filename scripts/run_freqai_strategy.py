#!/usr/bin/env python3
"""
Script to run the FreqAI optimized strategy
"""
import argparse
import logging
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('freqai_strategy.log')
    ]
)
logger = logging.getLogger('freqai_strategy')


def setup_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Run FreqAI optimized strategy')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/freqtrade_config.json',
        help='Path to FreqTrade configuration file'
    )
    
    parser.add_argument(
        '--strategy', 
        type=str, 
        default='FreqAIOptimizedStrategy',
        help='Strategy class name'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'live', 'dry_run'],
        default='dry_run',
        help='Trading mode: backtest, live, or dry_run'
    )
    
    parser.add_argument(
        '--timerange',
        type=str,
        default='',
        help='Timerange for backtesting (format: YYYYMMDD-YYYYMMDD)'
    )
    
    return parser.parse_args()


def run_strategy(args):
    """
    Run the FreqAI strategy with the specified arguments
    """
    # Import here to avoid circular imports
    import sys
    
    # Prepare FreqTrade arguments
    ft_args = [
        sys.argv[0],  # Script name
    ]
    
    # Add mode-specific command
    if args.mode == 'backtest':
        ft_args.append('backtesting')
    else:
        ft_args.append('trade')
    
    # Add common arguments
    ft_args.extend(['--config', args.config])
    ft_args.extend(['--strategy', args.strategy])
    ft_args.extend(['--freqaimodel', 'LightGBMRegressorMultiTarget'])
    
    # Add mode-specific arguments
    if args.mode == 'backtest':
        if args.timerange:
            ft_args.extend(['--timerange', args.timerange])
    elif args.mode == 'live':
        ft_args.append('--dry-run-wallet=false')
    
    # Run FreqTrade directly
    logger.info(f"Running FreqTrade with arguments: {' '.join(ft_args)}")
    
    # Replace sys.argv with our custom arguments
    sys.argv = ft_args
    
    # Import and run FreqTrade
    from freqtrade.main import main
    main()


def main():
    """
    Main function to run the FreqAI strategy
    """
    # Parse arguments
    args = setup_args()
    
    # Log the configuration
    logger.info(f"Running FreqAI strategy with config: {args.config}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Mode: {args.mode}")
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Run the strategy
    try:
        run_strategy(args)
    except Exception as e:
        logger.exception(f"Error running strategy: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 