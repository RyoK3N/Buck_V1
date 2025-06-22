from agent_scripts import __version__
import subprocess, sys


def test_version_constant():
    assert isinstance(__version__, str) and __version__


def test_cli_version():
    result = subprocess.run([sys.executable, '-m', 'agent_scripts.cli', '--version'], capture_output=True, text=True)
    assert result.returncode == 0
    assert __version__ in result.stdout.strip()
