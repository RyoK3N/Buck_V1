# Codex Version Roadmap

This document outlines four incremental versions of the Buck project that extend the existing `main` branch. Each version builds upon the previous one with improvements and new features.

## Version 1.0.0 – Baseline
This is the current state of the repository on the `main` branch. It provides:
- Asynchronous stock and news data retrieval
- Technical and sentiment analysis tools
- OpenAI-powered forecasting

## Version 1.1.0 – Improved Reliability
- Added missing Python dependencies in the development workflow
- Cleaned up unused imports in data providers
- Minor documentation fixes (e.g., notebook name correction)

## Version 1.2.0 – Enhanced Testing
- Introduced minimal test infrastructure with `pytest`
- Added `py_compile` step in CI to catch syntax errors
- Provided instructions for installing required packages

## Version 1.3.0 – Better Usability
- Clarified CLI examples in the README
- Standardized notebook file names for easier discovery
- Documented environment variable usage in more detail

## Version 1.4.0 – Future Ideas
- Potential integration with additional data sources
- Advanced ensemble forecasting strategies
- Support for configurable caching backends

These version names and notes can guide future development on a dedicated branch if desired.
