import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to import review.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from review import get_env_or_arg, validate_credentials


class TestReview(unittest.TestCase):
    def setUp(self):
        self.env_vars = {
            "OPENAI_API_KEY": "test_openai_key",
            "GITHUB_TOKEN": "test_github_token",
            "REPO_NAME": "test/repo",
            "PR_NUMBER": "1",
        }
        self.patcher = patch.dict(os.environ, self.env_vars)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_get_env_or_arg(self):
        # Test getting from environment
        self.assertEqual(get_env_or_arg("OPENAI_API_KEY", 1), "test_openai_key")

        # Test getting from args
        with patch("sys.argv", ["script.py", "arg_value"]):
            self.assertEqual(get_env_or_arg("NON_EXISTENT", 1), "arg_value")

        # Test default value
        self.assertEqual(get_env_or_arg("NON_EXISTENT", 1, "default"), "default")

    @patch("github.Github")
    @patch("openai.OpenAI")
    def test_validate_credentials_success(self, mock_openai, mock_github):
        # Setup mocks
        mock_github_instance = MagicMock()
        mock_github.return_value = mock_github_instance
        mock_github_instance.get_user.return_value.login = "test_user"

        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        mock_openai_instance.models.list.return_value = ["model1", "model2"]

        # Test successful validation
        self.assertTrue(validate_credentials("test_token", "test_key"))

    @patch("github.Github")
    def test_validate_credentials_github_failure(self, mock_github):
        # Setup mock to raise exception
        mock_github.side_effect = Exception("Invalid token")

        # Test failed validation
        self.assertFalse(validate_credentials("invalid_token", "test_key"))

    @patch("github.Github")
    @patch("openai.OpenAI")
    def test_validate_credentials_openai_failure(self, mock_openai, mock_github):
        # Setup GitHub mock for success
        mock_github_instance = MagicMock()
        mock_github.return_value = mock_github_instance
        mock_github_instance.get_user.return_value.login = "test_user"

        # Setup OpenAI mock for failure
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        mock_openai_instance.models.list.side_effect = Exception("Invalid key")

        # Test failed validation
        self.assertFalse(validate_credentials("test_token", "invalid_key"))


class TestCommentParsing(unittest.TestCase):
    def test_parse_review_comments(self):
        # Placeholder for future test implementation
        self.assertIsNone(None)  # Placeholder assertion


if __name__ == "__main__":
    unittest.main()
