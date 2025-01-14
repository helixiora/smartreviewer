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
        # Test GitHub token by accessing the repository
        g = Github(auth=Auth.Token(github_token))
        repo_name = get_required_env("REPO_NAME")
        _ = g.get_repo(repo_name).full_name
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

    logger.info("Configuration loaded:")
    logger.info("- Repository: %s", repo_name)
    logger.info("- PR Number: %s", pr_number)
    logger.info("- Model: %s", model)
    logger.info("- Review Type: %s", review_comment_type)
    logger.info("- Max Files: %d", max_files)
    logger.info("- Exclude Patterns: %s", exclude_patterns)

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
        logger.info("Successfully accessed repository: %s", repo.full_name)
        pr: PullRequest = repo.get_pull(pr_number)
        logger.info("Successfully accessed PR #%d: %s", pr_number, pr.title)
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
        files = list(pr.get_files())
        logger.info("Found %d files in PR", len(files))
        for file in files:
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
                logger.debug("Patch for %s: %s", file.filename, file.patch)
            else:
                logger.warning("No patch available for file: %s", file.filename)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error collecting code diffs: %s", str(e))
        sys.exit(1)

    logger.info("Collected snippets from %d files", files_reviewed)
    # Set files_reviewed output
    set_output("files_reviewed", str(files_reviewed))

    # Collect commit messages
    try:
        commits = list(pr.get_commits())
        logger.info("Found %d commits", len(commits))
        commit_messages = [commit.commit.message for commit in commits]
        logger.info("Commit messages: %s", commit_messages)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error collecting commit messages: %s", str(e))
        sys.exit(1)

    # Collect PR description
    pr_description = pr.body or ""
    logger.info("PR description length: %d characters", len(pr_description))

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
        logger.info("Received review response of length: %d", len(review_comments))
        logger.debug("Review response: %s", review_comments)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error getting review from OpenAI: %s", str(e))
        sys.exit(1)

    # Extract verdict and summary
    verdict = "commented"  # default
    summary = ""
    for line in review_comments.split("\n"):
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip().lower()
            logger.info("Review verdict: %s", verdict)
        elif line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()
            logger.info("Review summary length: %d", len(summary))

    # Set outputs
    set_output("review_result", verdict)
    set_output("review_summary", json.dumps(summary))

    # Post review based on type
    try:
        if review_comment_type == "summarized":
            logger.info("Posting summarized review")
            latest_commit = pr.get_commits().reversed[0]
            logger.info("Latest commit: %s", latest_commit.sha)
            # Post a single review with the summary
            pr.create_review(
                body=review_comments,
                commit=latest_commit.sha,
                event=verdict.upper()
                if verdict in ["approved", "changes_requested"]
                else "COMMENT",
            )
        else:
            logger.info("Posting individual review comments")
            # Parse and post individual comments
            current_file = None
            current_line = None
            current_comment = []
            review_body = []
            latest_commit = pr.get_commits().reversed[0]
            logger.info("Latest commit: %s", latest_commit.sha)

            for line in review_comments.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if line.startswith("FILE:"):
                    # If we have a previous comment, submit it
                    if current_file and current_line and current_comment:
                        comment_text = "\n".join(current_comment)
                        try:
                            logger.info("Posting comment for %s:%d", current_file, current_line)
                            pr.create_review_comment(
                                body=comment_text,
                                commit=latest_commit.sha,
                                path=current_file,
                                line=current_line,
                            )
                            logger.info("Successfully posted comment")
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
                    logger.info("Posting final comment for %s:%d", current_file, current_line)
                    pr.create_review_comment(
                        body=comment_text,
                        commit=latest_commit.sha,
                        path=current_file,
                        line=current_line,
                    )
                    logger.info("Successfully posted final comment")
                except Exception as e:
                    logger.warning(
                        "Failed to post comment on %s:%s: %s",
                        current_file,
                        current_line,
                        str(e),
                    )
                    review_body.append(f"{current_file}:{current_line} - {comment_text}")

            # Create the final review with verdict
            try:
                logger.info("Creating final review with verdict: %s", verdict)
                logger.info("Review body length: %d", len("\n\n".join(review_body)))
                pr.create_review(
                    body="\n\n".join(review_body),
                    commit=latest_commit.sha,
                    event=verdict.upper()
                    if verdict in ["approved", "changes_requested"]
                    else "COMMENT",
                )
                logger.info("Successfully posted final review")
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("Error posting review. Details: %s", str(e))
                logger.error("Review body: %s", "\n\n".join(review_body))
                logger.error("Verdict: %s", verdict)
                logger.error("Commit SHA: %s", latest_commit.sha)
                sys.exit(1)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error in review process: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
