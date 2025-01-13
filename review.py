#!/usr/bin/env python

import os
import sys
import logging
from openai import OpenAI
from github import Github
from github.GithubException import BadCredentialsException, UnknownObjectException
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv(override=True)

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_env_or_arg(env_var, arg_index, default=None):
    """Get value from environment variable or command line argument."""
    return os.getenv(env_var) or (
        sys.argv[arg_index] if len(sys.argv) > arg_index else default
    )


def validate_credentials(github_token, openai_api_key):
    """Validate GitHub and OpenAI credentials."""
    try:
        # Test GitHub token
        g = Github(github_token)
        g.get_user().login
        logger.info("GitHub credentials validated successfully")
    except BadCredentialsException:
        logger.error("Invalid GitHub token")
        return False

    try:
        # Test OpenAI API key
        client = OpenAI(api_key=openai_api_key)
        client.models.list()
        logger.info("OpenAI credentials validated successfully")
    except Exception as e:
        logger.error(f"Invalid OpenAI API key: {str(e)}")
        return False

    return True


def main():
    # Get configuration
    openai_api_key = get_env_or_arg("OPENAI_API_KEY", 1)
    github_token = get_env_or_arg("GITHUB_TOKEN", 2)
    repo_name = get_env_or_arg("REPO_NAME", 3)
    pr_number = get_env_or_arg("PR_NUMBER", 4)
    dry_run = get_env_or_arg("DRY_RUN", 5, "false").lower() == "true"

    if not all([openai_api_key, github_token, repo_name, pr_number]):
        logger.error(
            "Missing required environment variables or command line arguments. "
            "Please set them in .env file or provide as command line arguments."
        )
        sys.exit(1)

    try:
        pr_number = int(pr_number)
    except ValueError:
        logger.error("PR_NUMBER must be a valid integer")
        sys.exit(1)

    # Log configuration in debug mode
    logger.debug(f"Configuration loaded:")
    logger.debug(f"- Repository: {repo_name}")
    logger.debug(f"- PR Number: {pr_number}")
    logger.debug(f"- Dry Run: {dry_run}")

    # Validate credentials if not in dry-run mode
    if not dry_run and not validate_credentials(github_token, openai_api_key):
        sys.exit(1)

    # Check for dry-run mode
    if dry_run:
        logger.info("DRY RUN: Skipping GitHub API calls")
        logger.info("DRY RUN: Would review PR and generate the following comment:")
        print(
            "### OpenAI Code Review:\n\nThis is a dry run. In actual execution, this would contain the AI-generated review."
        )
        sys.exit(0)

    # Initialize GitHub client
    logger.info("Initializing GitHub client")
    g = Github(github_token)

    try:
        repo = g.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
    except UnknownObjectException:
        logger.error(f"Could not find PR #{pr_number} in repository {repo_name}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error accessing PR: {str(e)}")
        sys.exit(1)

    logger.info(f"Reviewing PR: {pr.title}")

    # Collect code diffs
    code_snippets = []
    try:
        for file in pr.get_files():
            logger.info(f"Collecting code snippets from: {file.filename}")
            if file.patch:
                code_snippets.append(file.patch)
    except Exception as e:
        logger.error(f"Error collecting code diffs: {str(e)}")
        sys.exit(1)

    # Collect commit messages
    try:
        commit_messages = [commit.commit.message for commit in pr.get_commits()]
    except Exception as e:
        logger.error(f"Error collecting commit messages: {str(e)}")
        sys.exit(1)

    # Collect PR description
    pr_description = pr.body or ""

    # Collect comments and reviews
    try:
        comments = pr.get_issue_comments()
        reviews = pr.get_reviews()

        all_comments = [comment.body for comment in comments] + [
            review.body for review in reviews if review.body
        ]
    except Exception as e:
        logger.error(f"Error collecting comments and reviews: {str(e)}")
        sys.exit(1)

    # Create the prompt for OpenAI
    prompt = f"""
Review the following pull request:
- Description: {pr_description}
- Commit Messages: {" ".join(commit_messages)}
- Code Snippets: {" ".join(code_snippets)}
- Comments and Reviews: {" ".join(all_comments)}

Provide a detailed review with suggestions for improvements.
"""

    # Request code review from OpenAI
    logger.info("Requesting code review from OpenAI")
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": 'You are a code review assistant helping a developer review a pull request. \
                        Don\'t acknowledge good code, only provide feedback on issues or improvements. \
                        Be specific and actionable in your feedback. Make sure that your comments are \
                        respectful and constructive. Provide links to relevant documentation if possible. \
                        When discussing code, refer to specific lines or sections of files whenever \
                        possible. Don\'t ask "consider doing x" or "maybe you could do y". Instead, say \
                        "do x" or "do y". Only provide the list of comments and suggestions, not the full \
                        review. Don\'t ask to refactor code unless it\'s necessary. \
                        Don\'t provide feedback on the PR description, commit messages, or \
                        comments/reviews.',
                },
                {"role": "user", "content": prompt},
            ],
        )
        review_comments = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting review from OpenAI: {str(e)}")
        sys.exit(1)

    # Post review comments back to PR
    comment_body = f"### OpenAI Code Review:\n\n{review_comments}"

    if dry_run:
        logger.info("DRY RUN: Would post the following comment:")
        print(comment_body)
    else:
        try:
            logger.info("Posting review comments to PR")
            pr.create_issue_comment(comment_body)
            logger.info("Review comments posted successfully")
        except Exception as e:
            logger.error(f"Error posting review comments: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    main()
