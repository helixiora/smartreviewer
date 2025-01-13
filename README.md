# AI Pull Request Reviewer

A GitHub Action that automatically reviews pull requests using AI to provide detailed feedback. The action uses OpenAI's GPT-4-Turbo model to analyze code changes and provide actionable suggestions for improvements.

## Features

- Analyzes code changes in pull requests
- Provides detailed, actionable feedback
- Reviews code style, potential bugs, and suggests improvements
- Integrates with OpenAI's GPT-4-Turbo model
- Robust error handling and credential validation
- Dry-run mode for testing

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
    steps:
      - name: AI Pull Request Review
        uses: helixiora/smartreviewer@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
          # Optional: Enable dry-run mode
          # dry_run: 'true'
```

## Action Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `github_token` | GitHub token for API access | Yes | N/A |
| `openai_api_key` | OpenAI API key | Yes | N/A |
| `repo_name` | Repository name (owner/repo) | No | Current repository |
| `pr_number` | Pull request number | No | Current PR number |
| `dry_run` | Run in dry-run mode (will not post comments) | No | 'false' |

## Local Development

### Prerequisites

- Python 3.11 or higher
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/helixiora/smartreviewer.git
cd smartreviewer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
```bash
cp .env.example .env
```

4. Configure your environment variables in `.env`:
```bash
cp .env.example .env
```

### Running Locally

Run the script directly:
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
| `GITHUB_TOKEN` | GitHub personal access token | Yes | N/A |
| `REPO_NAME` | Repository name in format owner/repo | Yes | N/A |
| `PR_NUMBER` | Pull request number to review | Yes | N/A |
| `DRY_RUN` | Run without posting comments | No | false |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No | INFO |

### Token Permissions

#### GitHub Token
The GitHub token needs the following permissions:
- `repo` scope for private repositories
- `public_repo` scope for public repositories
- Pull request read/write access

#### OpenAI API Key
The OpenAI API key needs access to:
- GPT-4-Turbo model
- Chat completions API

## Features in Detail

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
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT
