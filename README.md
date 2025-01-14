# Helixiora AI PR Reviewer

[![Version](https://img.shields.io/badge/version-1.2.4-blue.svg)](https://github.com/helixiora/smartreviewer/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Marketplace](https://img.shields.io/badge/Marketplace-Helixiora%20AI%20PR%20Reviewer-blue)](https://github.com/marketplace/actions/helixiora-ai-pr-reviewer)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A GitHub Action that automatically reviews pull requests using AI to provide detailed, actionable
feedback. The action uses OpenAI's GPT-4-Turbo model to analyze code changes and provide specific,
line-by-line suggestions for improvements.

## Features

- Provides inline code review comments on specific lines
- Focuses on actionable, high-quality feedback
- Only suggests concrete, implementable improvements
- Analyzes code for security, bugs, performance, and maintainability
- Integrates with OpenAI's GPT-4-Turbo model
- Includes dry-run mode for testing
- Automatic GitHub token handling

## Review Quality

The reviewer is designed to:

- Focus on quality over quantity
- Only provide specific, actionable feedback
- Suggest concrete improvements rather than vague advice
- Point out meaningful issues that affect code quality
- Accept that well-written code may need few or no changes

Example of a review comment:

```text
Add error handling for the database connection. Wrap the connection attempt in a try-catch block and implement a retry mechanism with exponential backoff.
```

## GitHub Action Usage

Add this action to your workflow:

```yaml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write  # Required for posting review comments
    steps:
      - name: AI Pull Request Review
        uses: helixiora/smartreviewer@v1
        with:
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
          # Optional: Enable dry-run mode
          # dry_run: 'true'
```

## Configuration

### Action Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `openai_api_key` | OpenAI API key | Yes | N/A |
| `repo_name` | Repository name (owner/repo) | No | Current repository |
| `pr_number` | Pull request number | No | Current PR number |
| `dry_run` | Run in dry-run mode (will not post comments) | No | 'false' |

### GitHub Token Handling

The action automatically uses the `GITHUB_TOKEN` provided by GitHub Actions to:

- Read the pull request content
- Post inline review comments
- Access repository information

Make sure to set the appropriate permissions in your workflow:

```yaml
permissions:
  pull-requests: write
```

## Local Development

### Prerequisites

- Python 3.11 or higher
- Docker (optional)
- GitHub Personal Access Token (for local testing only)
- pre-commit (for development)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/helixiora/smartreviewer.git
    cd smartreviewer
    ```

2. Create and activate a virtual environment:

    ```bash
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up pre-commit hooks:

    ```bash
    pre-commit install
    ```

5. Create a `.env` file:

    ```bash
    cp .env.example .env
    ```

6. Configure your environment variables in `.env`:

    ```ini
    # Required settings
    OPENAI_API_KEY=your_openai_api_key_here
    GITHUB_TOKEN=your_github_token_here  # Only needed for local development

    # Optional settings
    REPO_NAME=owner/repo
    PR_NUMBER=1

    # Debug settings
    DRY_RUN=false
    LOG_LEVEL=INFO
    ```

### Running Locally

Make sure your virtual environment is activated:

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
.\venv\Scripts\activate
```

Then run the script:

```bash
python review.py
```

Or with Docker:

```bash
# Build the image
docker build -t smartreviewer .

# Run with .env file
docker run --rm --env-file .env smartreviewer

# Or run with environment variables
docker run --rm \
  -e OPENAI_API_KEY=your_key \
  -e GITHUB_TOKEN=your_token \
  -e REPO_NAME=owner/repo \
  -e PR_NUMBER=1 \
  smartreviewer
```

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes | N/A |
| `GITHUB_TOKEN` | GitHub personal access token (local dev only) | Yes* | N/A |
| `REPO_NAME` | Repository name in format owner/repo | Yes | N/A |
| `PR_NUMBER` | Pull request number to review | Yes | N/A |
| `DRY_RUN` | Run without posting comments | No | false |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No | INFO |

*Only required for local development, not needed in GitHub Actions

### Token Permissions

#### GitHub Token

When running locally, the GitHub token needs the following permissions:

- `repo` scope for private repositories
- `public_repo` scope for public repositories
- Pull request read/write access

In GitHub Actions, these permissions are handled automatically through the workflow permissions.

#### OpenAI API Key

The OpenAI API key needs access to:

- GPT-4-Turbo model
- Chat completions API

## Features in Detail

### Review Comments

The action posts review comments that are:

- Tied to specific lines in the code
- Part of a unified review
- Focused on actionable improvements
- Clear and specific in their suggestions

### Error Handling

The action includes comprehensive error handling for:

- Invalid credentials
- Missing or malformed inputs
- GitHub API errors
- OpenAI API errors
- Network issues

All errors are logged with clear messages to help with troubleshooting.

### Dry Run Mode

Enable dry-run mode to test the action without posting comments to your pull requests:

```yaml
with:
  dry_run: 'true'
```

In dry-run mode, the action will:

- Validate all inputs and credentials
- Process the pull request
- Generate the review
- Print the review to the logs instead of posting it

### Logging

Configure the logging level using the `LOG_LEVEL` environment variable:

- `DEBUG`: Detailed information for debugging
- `INFO`: General information about progress (default)
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages only

## Contributing

1. Fork the repository
2. Create a feature branch
3. Install development dependencies and pre-commit hooks:

    ```bash
    pip install -r requirements.txt
    pre-commit install
    ```

4. Make your changes
5. Ensure all tests pass and pre-commit checks succeed:

    ```bash
    pytest
    pre-commit run --all-files
    ```

6. Commit your changes (pre-commit hooks will run automatically)
7. Push to your branch
8. Create a Pull Request

### Pre-commit Hooks

The following checks run automatically on each commit:

- Code formatting (ruff)
- Linting (pylint)
- Security checks (bandit)
- YAML/TOML validation
- Markdown linting
- Type checking (mypy)
- Various file checks (trailing whitespace, merge conflicts, etc.)

To run checks manually:

```bash
pre-commit run --all-files
```

To update hooks to their latest versions:

```bash
pre-commit autoupdate
```

## License

MIT
