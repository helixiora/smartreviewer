#!/usr/bin/env python

import fnmatch
import json
import logging
import os
import sys
from typing import TYPE_CHECKING, List

from dotenv import load_dotenv
from github import Auth, Github
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


def get_required_env(env_var):
    """Get required environment variable."""
    value = os.getenv(env_var)
    if not value:
        logger.error("Missing required environment variable: %s", env_var)
        sys.exit(1)
    return value


def get_optional_env(env_var, default=None):
    """Get optional environment variable with default value."""
    return os.getenv(env_var, default)


def validate_credentials(github_token, openai_api_key):
    """Validate GitHub and OpenAI credentials."""
    try:
        # Test GitHub token
        g = Github(auth=Auth.Token(github_token))
        _ = g.get_user().login
        logger.info("GitHub credentials validated successfully")
    except BadCredentialsException:
        logger.error("Invalid GitHub token")
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Unexpected error during GitHub validation: %s", str(e))
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


def should_exclude_file(filename: str, exclude_patterns: List[str]) -> bool:
    """Check if a file should be excluded based on patterns."""
    if not exclude_patterns:
        return False

    return any(
        fnmatch.fnmatch(filename, pattern.strip())
        for pattern in exclude_patterns
        if pattern.strip()
    )


def set_output(name: str, value: str):
    """Set an output variable for GitHub Actions."""
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def main():
    # Get required configuration
    openai_api_key = get_required_env("OPENAI_API_KEY")
    github_token = get_required_env("GITHUB_TOKEN")
    repo_name = get_required_env("REPO_NAME")
    pr_number = get_required_env("PR_NUMBER")

    # Get optional configuration
    model = get_optional_env("MODEL", "gpt-4")
    review_comment_type = get_optional_env("REVIEW_COMMENT_TYPE", "individual")
    max_files = int(get_optional_env("MAX_FILES", "50"))
    exclude_patterns = get_optional_env("EXCLUDE_PATTERNS", "").split("\n")

    try:
        pr_number = int(pr_number)
    except ValueError:
        logger.error("PR_NUMBER must be a valid integer")
        sys.exit(1)

    # Validate credentials
    if not validate_credentials(github_token, openai_api_key):
        sys.exit(1)

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
    files_reviewed = 0
    try:
        for file in pr.get_files():
            if files_reviewed >= max_files:
                logger.warning(
                    "Reached maximum file limit (%d). Skipping remaining files.", max_files
                )
                break

            if should_exclude_file(file.filename, exclude_patterns):
                logger.info("Skipping excluded file: %s", file.filename)
                continue

            logger.info("Collecting code snippets from: %s", file.filename)
            if file.patch:
                code_snippets.append(f"File: {file.filename}\n{file.patch}")
                files_reviewed += 1
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error collecting code diffs: %s", str(e))
        sys.exit(1)

    # Set files_reviewed output
    set_output("files_reviewed", str(files_reviewed))

    # Collect commit messages
    try:
        commit_messages = [commit.commit.message for commit in pr.get_commits()]
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error collecting commit messages: %s", str(e))
        sys.exit(1)

    # Collect PR description
    pr_description = pr.body or ""

    # Create the prompt for OpenAI based on review type
    if review_comment_type == "summarized":
        prompt = (
            "Review the following pull request and provide a comprehensive summary of all changes "
            "and potential improvements. Focus on high-level patterns and architectural concerns:\n"
            f"PR Title: {pr.title}\n"
            f"Description: {pr_description}\n"
            f"Files Changed: {files_reviewed}\n"
            f"Commit Messages:\n{chr(10).join(f'- {msg}' for msg in commit_messages)}\n\n"
            "Code Changes:\n"
            f"{chr(10).join(code_snippets)}\n\n"
            "Provide your review in the following format:\n"
            "SUMMARY: Overall assessment of the changes\n"
            "IMPROVEMENTS: List of suggested improvements\n"
            "CONCERNS: Any concerns or potential issues\n"
            "VERDICT: Either 'approved', 'commented', or 'changes_requested'"
        )
    else:  # individual comments
        prompt = (
            "Review the following pull request and provide specific, actionable feedback "
            "for each issue found:\n\n"
            f"PR Title: {pr.title}\n"
            f"Description: {pr_description}\n"
            f"Commit Messages:\n{chr(10).join(f'- {msg}' for msg in commit_messages)}\n\n"
            "Code Changes:\n"
            f"{chr(10).join(code_snippets)}\n\n"
            "For each issue, provide:\n"
            "FILE: <file_path>\n"
            "LINE: <line_number_or_range>\n"
            "COMMENT: <specific_actionable_feedback>\n\n"
            "End your review with:\n"
            "VERDICT: Either 'approved', 'commented', or 'changes_requested'\n"
            "SUMMARY: Brief overview of your findings"
        )

    # Request code review from OpenAI
    logger.info("Requesting code review from OpenAI using model: %s", model)
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a code review assistant helping a developer review a pull request."
                        "Focus on providing high-quality, actionable feedback. "
                        "It's fine to approve if the code is well-written. "
                        "Each comment must include specific, concrete suggestions for improvement. "
                        "Don't use weak language - be direct and clear."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        review_comments = response.choices[0].message.content
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error getting review from OpenAI: %s", str(e))
        sys.exit(1)

    # Extract verdict and summary
    verdict = "commented"  # default
    summary = ""
    for line in review_comments.split("\n"):
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip().lower()
        elif line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()

    # Set outputs
    set_output("review_result", verdict)
    set_output("review_summary", json.dumps(summary))

    # Post review based on type
    try:
        if review_comment_type == "summarized":
            # Post a single review with the summary
            pr.create_review(
                body=review_comments,
                commit=pr.get_commits().reversed[0].sha,
                event=verdict.upper()
                if verdict in ["approved", "changes_requested"]
                else "COMMENT",
            )
        else:
            # Parse and post individual comments
            current_file = None
            current_line = None
            current_comment = []
            review_body = []

            for line in review_comments.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if line.startswith("FILE:"):
                    # If we have a previous comment, submit it
                    if current_file and current_line and current_comment:
                        comment_text = "\n".join(current_comment)
                        try:
                            pr.create_review_comment(
                                body=comment_text,
                                commit=pr.get_commits().reversed[0].sha,
                                path=current_file,
                                line=current_line,
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to post comment on %s:%s: %s",
                                current_file,
                                current_line,
                                str(e),
                            )
                            review_body.append(f"{current_file}:{current_line} - {comment_text}")

                    current_file = line.replace("FILE:", "").strip()
                    current_comment = []
                elif line.startswith("LINE:"):
                    current_line = int(line.replace("LINE:", "").strip().split("-")[0])
                elif line.startswith("COMMENT:"):
                    current_comment.append(line.replace("COMMENT:", "").strip())
                elif line.startswith(("VERDICT:", "SUMMARY:")):
                    review_body.append(line)

            # Submit the last comment if there is one
            if current_file and current_line and current_comment:
                comment_text = "\n".join(current_comment)
                try:
                    pr.create_review_comment(
                        body=comment_text,
                        commit=pr.get_commits().reversed[0].sha,
                        path=current_file,
                        line=current_line,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to post comment on %s:%s: %s",
                        current_file,
                        current_line,
                        str(e),
                    )
                    review_body.append(f"{current_file}:{current_line} - {comment_text}")

            # Create the final review with verdict
            pr.create_review(
                body="\n\n".join(review_body),
                commit=pr.get_commits().reversed[0].sha,
                event=verdict.upper()
                if verdict in ["approved", "changes_requested"]
                else "COMMENT",
            )

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error posting review: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
