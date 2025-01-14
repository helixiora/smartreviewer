#!/usr/bin/env python

import logging
import os
import sys
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from github import Github
from github.GithubException import BadCredentialsException, UnknownObjectException
from openai import OpenAI

if TYPE_CHECKING:
    from github.PullRequest import PullRequest
    from github.Repository import Repository

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
    return os.getenv(env_var) or (sys.argv[arg_index] if len(sys.argv) > arg_index else default)


def validate_credentials(github_token, openai_api_key):
    """Validate GitHub and OpenAI credentials."""
    try:
        # Test GitHub token
        g = Github(github_token)
        _ = g.get_user().login  # Assign to _ to avoid W0106
        logger.info("GitHub credentials validated successfully")
    except BadCredentialsException:
        logger.error("Invalid GitHub token")
        return False

    try:
        # Test OpenAI API key
        client = OpenAI(api_key=openai_api_key)
        client.models.list()
        logger.info("OpenAI credentials validated successfully")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Invalid OpenAI API key: %s", str(e))
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
        error_msg = (
            "Missing required environment variables or command line arguments. "
            "Please set them in .env file or provide as command line arguments."
        )
        logger.error(error_msg)
        sys.exit(1)

    try:
        pr_number = int(pr_number)
    except ValueError:
        logger.error("PR_NUMBER must be a valid integer")
        sys.exit(1)

    # Log configuration in debug mode
    logger.debug("Configuration loaded:")
    logger.debug("- Repository: %s", repo_name)
    logger.debug("- PR Number: %s", pr_number)
    logger.debug("- Dry Run: %s", dry_run)

    # Validate credentials if not in dry-run mode
    if not dry_run and not validate_credentials(github_token, openai_api_key):
        sys.exit(1)

    # Check for dry-run mode
    if dry_run:
        logger.info("DRY RUN: Skipping GitHub API calls")
        logger.info("DRY RUN: Would review PR and generate the following comment:")
        print(
            "### OpenAI Code Review:\n\nThis is a dry run. "
            "In actual execution, this would contain the AI-generated review."
        )
        sys.exit(0)

    # Initialize GitHub client
    logger.info("Initializing GitHub client")
    g = Github(github_token)

    try:
        repo: Repository = g.get_repo(repo_name)
        pr: PullRequest = repo.get_pull(pr_number)
    except UnknownObjectException:
        logger.error("Could not find PR #%s in repository %s", pr_number, repo_name)
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error accessing PR: %s", str(e))
        sys.exit(1)

    logger.info("Reviewing PR: %s", pr.title)

    # Collect code diffs
    code_snippets = []
    try:
        for file in pr.get_files():
            logger.info("Collecting code snippets from: %s", file.filename)
            if file.patch:
                code_snippets.append(file.patch)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error collecting code diffs: %s", str(e))
        sys.exit(1)

    # Collect commit messages
    try:
        commit_messages = [commit.commit.message for commit in pr.get_commits()]
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error collecting commit messages: %s", str(e))
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
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error collecting comments and reviews: %s", str(e))
        sys.exit(1)

    # Create the prompt for OpenAI
    prompt = (
        "Review the following pull request and provide ONLY actionable feedback "
        "for clear issues or improvements:\n"
        f"- Description: {pr_description}\n"
        f"- Commit Messages: {' '.join(commit_messages)}\n"
        f"- Code Snippets: {' '.join(code_snippets)}\n"
        f"- Comments and Reviews: {' '.join(all_comments)}\n\n"
        "Focus on quality over quantity. Only point out issues that:\n"
        "1. Have a clear, actionable solution\n"
        "2. Would meaningfully improve the code\n"
        "3. Address actual problems (security, bugs, performance, maintainability)\n\n"
        "It's perfectly acceptable to find no issues or just a few issues if the code is \
well-written.\n\n"
        "For each actionable issue you find, specify:\n"
        "1. The exact file path\n"
        "2. The specific line number or range from the diff\n"
        "3. A clear, concrete suggestion for improvement\n\n"
        "Format each comment as:\n"
        "FILE: <file_path>\n"
        "LINE: <line_number_or_range>\n"
        "COMMENT: <specific_actionable_feedback>\n\n"
        "Example of a good comment:\n"
        "FILE: src/main.py\n"
        "LINE: 45\n"
        "COMMENT: Add error handling for the database connection. Wrap the connection attempt in a "
        "try-catch block and implement a retry mechanism with exponential backoff.\n\n"
        "Example of a comment that is NOT actionable (avoid these):\n"
        "FILE: src/main.py\n"
        "LINE: 45\n"
        "COMMENT: This code could be better. Consider improving the error handling."
    )

    # Request code review from OpenAI
    logger.info("Requesting code review from OpenAI")
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a code review assistant helping a developer review a pull request."
                        "Focus on providing only high-quality, actionable feedback. "
                        "It's perfectly fine to find no issues if the code is well-written. "
                        "Each comment must include specific, concrete suggestions for improvement. "
                        'Don\'t use phrases like "consider", "maybe", or "might want to". '
                        'Instead, give direct, actionable advice like "Add", "Remove", \
"Replace with", etc. '
                        "Don't comment on style unless it violates a clear best practice. "
                        "Don't suggest refactoring unless there's a clear maintainability issue. "
                        "Don't provide feedback on PR descriptions or commit messages. "
                        "Format each comment with FILE:, LINE:, and COMMENT: prefixes."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        review_comments = response.choices[0].message.content
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error getting review from OpenAI: %s", str(e))
        sys.exit(1)

    # Parse review comments and post them as review comments
    if dry_run:
        logger.info("DRY RUN: Would post the following comments:")
        print(review_comments)
    else:
        try:
            logger.info("Posting review comments")

            # Create a new review
            pr.create_review(
                commit=pr.get_commits().reversed[0].sha,
                body="AI Code Review Comments",
                event="COMMENT",
            )

            # Parse and post individual comments
            current_file = None
            current_line = None
            current_comment = []

            for line in review_comments.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if line.startswith("FILE:"):
                    # If we have a previous comment, submit it
                    if current_file and current_line and current_comment:
                        comment_text = "\n".join(current_comment)
                        try:
                            logger.debug("Posting comment on %s:%s", current_file, current_line)
                            pr.create_review_comment(
                                body=comment_text,
                                commit=pr.get_commits().reversed[0].sha,
                                path=current_file,
                                line=current_line,
                            )
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            logger.warning(
                                "Failed to post comment on %s:%s: %s",
                                current_file,
                                current_line,
                                str(e),
                            )

                    current_file = line.replace("FILE:", "").strip()
                    current_comment = []
                elif line.startswith("LINE:"):
                    current_line = int(line.replace("LINE:", "").strip().split("-")[0])
                elif line.startswith("COMMENT:"):
                    current_comment.append(line.replace("COMMENT:", "").strip())

            # Submit the last comment if there is one
            if current_file and current_line and current_comment:
                comment_text = "\n".join(current_comment)
                try:
                    logger.debug("Posting comment on %s:%s", current_file, current_line)
                    pr.create_review_comment(
                        body=comment_text,
                        commit=pr.get_commits().reversed[0].sha,
                        path=current_file,
                        line=current_line,
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.warning(
                        "Failed to post comment on %s:%s: %s",
                        current_file,
                        current_line,
                        str(e),
                    )

            # Submit the review
            try:
                pr.create_review(
                    commit=pr.get_commits().reversed[0].sha,
                    body="AI Code Review Complete",
                    event="COMMENT",
                )
                logger.info("Review submitted successfully")
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Error submitting review: %s", str(e))
                sys.exit(1)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error posting review comments: %s", str(e))
            sys.exit(1)


if __name__ == "__main__":
    main()
