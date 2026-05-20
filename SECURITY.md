# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Buck, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, email the maintainers directly. You can find contact information in the repository's GitHub profile. We will acknowledge your report within 48 hours and work with you to understand and address the issue.

Please include:
- A description of the vulnerability
- Steps to reproduce it
- The potential impact
- Any suggested fixes (if you have them)

---

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x (current) | Yes |
| < 1.0 | No |

---

## Security Considerations

### API Keys and Secrets

- **Never commit API keys** to the repository. The `.gitignore` file excludes `.env` files.
- API keys are loaded from environment variables via Pydantic settings (`agent_scripts/config.py`).
- The `.env.example` file contains placeholder values only — copy it to `.env` and fill in your actual keys.
- If you accidentally commit a key, rotate it immediately and use `git filter-branch` or BFG Repo Cleaner to remove it from history.

### LLM Input/Output Logging

- The application saves LLM prompts and analysis inputs to the `inputs/` directory for debugging purposes.
- These files may contain your API configuration metadata (model name, temperature) but **not** the API key itself.
- Review the `inputs/` directory before sharing your project folder with others.
- The `output/` directory contains analysis results as JSON files. These are not sensitive but may contain financial data.

### Web-Facing Components

- The FastAPI backend runs with CORS configured for `localhost` origins only. If you deploy to production, restrict CORS to your actual domain.
- The backend accepts API keys in POST request bodies (for the UI to pass keys per-request). In production, use server-side environment variables instead of passing keys from the frontend.
- There is no built-in authentication on the API endpoints. If you expose the backend to the internet, add an authentication layer (API key middleware, OAuth, etc.).

### Dependencies

- Dependencies are pinned by major version in `requirements.txt`. Run `pip audit` periodically to check for known vulnerabilities.
- The frontend uses npm packages — run `npm audit` in `UI/frontend/` to check for known issues.
- PyTorch is included for the LSTM tool. It is a large dependency; if you don't need deep learning tools, you can remove it from `requirements.txt` (the maths tools will still work).

### Data Handling

- Stock data is fetched from Yahoo Finance and cached in-memory during a session. It is not persisted unless `save_results=True` is passed.
- News data may be fetched from third-party APIs. We do not control the content or availability of these sources.
- No user data is collected, stored, or transmitted to any service other than the configured LLM API.

### Tool Execution

- Tools execute arbitrary computations on stock data (mathematical calculations, ML model training, neural network inference). All of this runs locally.
- Web tools (planned, in `tools/web/`) will make outbound HTTP requests to public APIs (Yahoo Finance, SEC EDGAR, Reddit). These requests are read-only.
- The dynamic tool loader (`ToolFactory`) imports Python modules from the `tools/` directory. Only place trusted code in this directory.

---

## Best Practices for Deployment

1. **Use environment variables** for all secrets — never hardcode them.
2. **Restrict CORS** to your actual frontend domain.
3. **Add authentication** to API endpoints if exposed beyond localhost.
4. **Run behind a reverse proxy** (nginx, Caddy) with HTTPS in production.
5. **Pin exact dependency versions** in production (`pip freeze > requirements.lock`).
6. **Regularly update dependencies** and check for CVEs with `pip audit` and `npm audit`.
7. **Don't expose the `inputs/` or `output/` directories** through your web server.
