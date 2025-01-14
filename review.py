#!/usr/bin/env python

import fnmatch
import json
import logging
import os
import sys
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from dotenv import load_dotenv
from github import Auth, Github
from github.GithubException import BadCredentialsException, UnknownObjectException
from openai import OpenAI
from pydantic import BaseModel, Field

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


class ReviewVerdict(str, Enum):
    """Possible review verdicts."""

    APPROVED = "approved"
    COMMENTED = "commented"
    CHANGES_REQUESTED = "changes_requested"

    @classmethod
    def values(cls) -> List[str]:
        """Get all possible values."""
        return [v.value for v in cls]


class ReviewComment(BaseModel):
    """Individual review comment."""

    file: str = Field(..., description="The file path being commented on")
    line: int = Field(..., description="The line number being commented on")
    comment: str = Field(
        ...,
        description="The review comment with specific, actionable feedback",
    )


class IndividualReview(BaseModel):
    """Individual review response format."""

    comments: List[ReviewComment] = Field(
        default_factory=list, description="List of review comments"
    )
    verdict: ReviewVerdict = Field(
        default=ReviewVerdict.COMMENTED, description="The overall verdict of the review"
    )
    summary: str = Field(default="", description="A brief overview of the changes and their impact")

    def to_schema(self) -> dict:
        """Convert to OpenAI schema format."""
        return {
            "type": "object",
            "properties": {
                "comments": {
                    "type": "array",
                    "description": "List of review comments",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": "The file path being commented on",
                            },
                            "line": {
                                "type": "integer",
                                "description": "The line number being commented on",
                            },
                            "comment": {
                                "type": "string",
                                "description": "The review comment with specific, actionable \
feedback",
                            },
                        },
                        "required": ["file", "line", "comment"],
                    },
                },
                "verdict": {
                    "type": "string",
                    "enum": ReviewVerdict.values(),
                    "description": "The overall verdict of the review",
                },
                "summary": {
                    "type": "string",
                    "description": "A brief overview of the changes and their impact",
                },
            },
            "required": ["comments", "verdict", "summary"],
        }


class SummarizedReview(BaseModel):
    """Summarized review response format."""

    summary: str = Field(
        default="", description="Overall assessment of the changes and their impact"
    )
    improvements: List[str] = Field(
        default_factory=list,
        description="List of specific, actionable improvements suggested",
    )
    concerns: List[str] = Field(
        default_factory=list,
        description="List of concerns or potential issues identified",
    )
    verdict: ReviewVerdict = Field(
        default=ReviewVerdict.COMMENTED,
        description="The overall verdict of the review",
    )

    def to_schema(self) -> dict:
        """Convert to OpenAI schema format."""
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Overall assessment of the changes and their impact",
                },
                "improvements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific, actionable improvements suggested",
                },
                "concerns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of concerns or potential issues identified",
                },
                "verdict": {
                    "type": "string",
                    "enum": ReviewVerdict.values(),
                    "description": "The overall verdict of the review",
                },
            },
            "required": ["summary", "improvements", "concerns", "verdict"],
        }


def format_review_body(
    review: IndividualReview | SummarizedReview, failed_comments: Optional[List[str]] = None
) -> str:
    """Format the review body based on the review type."""
    if isinstance(review, SummarizedReview):
        return (
            f"# Review Summary\n\n{review.summary}\n\n"
            "## Suggested Improvements\n\n"
            + "\n".join(f"- {improvement}" for improvement in review.improvements)
            + "\n\n## Concerns\n\n"
            + "\n".join(f"- {concern}" for concern in review.concerns)
        )

    body = f"# Review Summary\n\n{review.summary}"
    if review.comments:
        body += "\n\n## Review Comments\n\n" + "\n".join(
            f"- {comment.file}:{comment.line} - {comment.comment}" for comment in review.comments
        )
    elif review.verdict == ReviewVerdict.APPROVED:
        body += "\n\n✅ Code looks good! No issues found."
    if failed_comments:
        body += "\n\n## Failed Comments\n\n" + "\n".join(failed_comments)
    return body


class ReviewType(str, Enum):
    """Review comment types."""

    INDIVIDUAL = "individual"
    SUMMARIZED = "summarized"

    @classmethod
    def values(cls) -> List[str]:
        """Get all possible values."""
        return [v.value for v in cls]


def get_required_env(env_var: str) -> str:
    """Get required environment variable."""
    value = os.getenv(env_var)
    if not value:
        logger.error("Missing required environment variable: %s", env_var)
        sys.exit(1)
    return value


def get_optional_env(env_var: str, default: Optional[str] = None) -> str:
    """Get optional environment variable with default value."""
    return os.getenv(env_var) or (default or "")


def validate_credentials(github_token: str, openai_api_key: str) -> bool:
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


def set_output(name: str, value: str) -> None:
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
    model = get_optional_env("MODEL", "gpt-4o")
    review_comment_type = get_optional_env("REVIEW_COMMENT_TYPE", ReviewType.INDIVIDUAL.value)
    max_files = int(get_optional_env("MAX_FILES", "50"))
    exclude_patterns = get_optional_env("EXCLUDE_PATTERNS", "").split("\n")

    # Validate review type
    try:
        review_type = ReviewType(review_comment_type)
    except ValueError:
        logger.error(
            "Invalid review type: %s. Must be one of: %s",
            review_comment_type,
            ", ".join(ReviewType.values()),
        )
        sys.exit(1)

    logger.info("Configuration loaded:")
    logger.info("- Repository: %s", repo_name)
    logger.info("- PR Number: %s", pr_number)
    logger.info("- Model: %s", model)
    logger.info("- Review Type: %s", review_type.value)
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

    # Request code review from OpenAI
    logger.info("Requesting code review from OpenAI using model: %s", model)
    try:
        client = OpenAI(api_key=openai_api_key)

        if review_type == ReviewType.INDIVIDUAL:
            review_class = IndividualReview
        else:
            review_class = SummarizedReview

        # Create an empty instance just for schema generation
        json_schema = review_class().to_schema()

        prompt = (
            "Review the following pull request and provide structured feedback. "
            "Focus on providing specific, actionable feedback that helps improve code quality. "
            "Approve if the code is well-written, but always suggest improvements where possible. "
            "Be direct and clear in your feedback.\n\n"
            f"PR Title: {pr.title}\n"
            f"Description: {pr_description}\n"
            f"Files Changed: {files_reviewed}\n"
            f"Commit Messages:\n{chr(10).join([f'- {msg}' for msg in commit_messages])}\n\n"
            "Code Changes:\n"
            f"{chr(10).join(code_snippets)}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a code review assistant. Provide a thorough code review that is "
                        "specific, actionable, and helps improve code quality. Focus on concrete "
                        "suggestions and clear feedback."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object", "schema": json_schema},
        )

        # Extract the review from the structured output
        review_data = json.loads(response.choices[0].message.content)
        logger.info("Received structured review response")
        logger.debug("Review data: %s", review_data)

        # Parse and validate the review data
        try:
            if review_type == ReviewType.INDIVIDUAL:
                review = IndividualReview(**review_data)
            else:
                review = SummarizedReview(**review_data)
            logger.info("Successfully validated review data")
        except Exception as e:
            logger.error("Failed to validate review data: %s", str(e))
            sys.exit(1)

        # Set outputs
        set_output("review_result", review.verdict.value)
        set_output("review_summary", json.dumps(review.summary))

        # Post review based on type
        try:
            latest_commit = pr.get_commits().reversed[0]
            logger.info("Latest commit: %s", latest_commit.sha)
            review_event = (
                str(review.verdict).upper()
                if review.verdict != ReviewVerdict.COMMENTED
                else "COMMENT"
            )
            logger.info("Review event: %s", review_event)

            if review_type == ReviewType.SUMMARIZED:
                logger.info("Posting summarized review")
                pr.create_review(
                    body=format_review_body(review),
                    commit=latest_commit.sha,
                    event=review_event,
                )
            else:
                logger.info("Posting individual review comments")
                # Post individual comments
                failed_comments = []

                for comment in review.comments:
                    try:
                        logger.info("Posting comment for %s:%d", comment.file, comment.line)
                        pr.create_review_comment(
                            body=comment.comment,
                            commit=latest_commit.sha,
                            path=comment.file,
                            line=comment.line,
                        )
                        logger.info("Successfully posted comment")
                    except Exception as e:
                        logger.warning(
                            "Failed to post comment on %s:%s: %s",
                            comment.file,
                            comment.line,
                            str(e),
                        )
                        failed_comments.append(f"{comment.file}:{comment.line} - {comment.comment}")

                # Create the final review with verdict
                try:
                    logger.info("Creating final review with verdict: %s", review.verdict)
                    body = format_review_body(review, failed_comments)
                    logger.info("Review body length: %d", len(body))
                    pr.create_review(
                        body=body,
                        commit=latest_commit.sha,
                        event=review_event,
                    )
                    logger.info("Successfully posted final review")
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error("Error posting review. Details: %s", str(e))
                    logger.error("Review body: %s", body)
                    logger.error("Verdict: %s", review.verdict)
                    logger.error("Review event: %s", review_event)
                    logger.error("Commit SHA: %s", latest_commit.sha)
                    sys.exit(1)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error in review process: %s", str(e))
            sys.exit(1)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error getting review from OpenAI: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
