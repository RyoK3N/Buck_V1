import asyncio
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_scripts.data_providers import DataProviderFactory

@pytest.mark.asyncio
async def test_yahoo_finance_provider():
    provider = DataProviderFactory.create_yahoo_finance_provider()
    data = await provider.get_stock_data('BHEL.NS', '2024-01-01', '2024-01-10', '1d')
    assert data is not None
    assert 'data' in data and not data['data'].empty
