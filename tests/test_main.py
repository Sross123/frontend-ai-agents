"""Tests for the AI Multi-Agent Studio.

Run with: pytest
Or with coverage: pytest --cov=app tests/
"""

import pytest
import os
import tempfile
from pathlib import Path
from app.tools import validate_workspace_path, FileSystemSecurityError, write_to_file, read_file, list_files


class TestFileSystemSecurity:
    """Test path traversal protection."""

    def test_validate_workspace_path_allowed(self):
        """Test that valid paths within workspace are accepted."""
        base = "/home/user/workspace"
        result = validate_workspace_path(base, "file.html")
        assert result == "/home/user/workspace/file.html"

    def test_validate_workspace_path_traversal_blocked(self):
        """Test that directory traversal attempts are blocked."""
        base = "/home/user/workspace"
        with pytest.raises(FileSystemSecurityError):
            validate_workspace_path(base, "../../../etc/passwd")

    def test_validate_workspace_path_absolute_escape(self):
        """Test that absolute paths outside workspace are blocked."""
        base = "/home/user/workspace"
        with pytest.raises(FileSystemSecurityError):
            validate_workspace_path(base, "/etc/passwd")


class TestFileOperations:
    """Test file read/write operations."""

    @pytest.mark.asyncio
    async def test_write_and_read_file(self):
        """Test writing and reading files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_content = "<html><body>Hello</body></html>"

            # Write file
            result = await write_to_file(
                path="index.html",
                content=test_content,
                workspace_dir=tmpdir
            )
            assert "successfully" in result.lower()

            # Read file
            read_result = await read_file(
                path="index.html",
                workspace_dir=tmpdir
            )
            assert read_result == test_content

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await read_file(
                path="nonexistent.html",
                workspace_dir=tmpdir
            )
            assert "does not exist" in result

    @pytest.mark.asyncio
    async def test_list_files(self):
        """Test listing files in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            Path(tmpdir, "file1.html").write_text("<html></html>")
            Path(tmpdir, "file2.css").write_text("body { }")

            result = await list_files(
                directory=".",
                workspace_dir=tmpdir
            )

            assert len(result) >= 2
            assert "file1.html" in result
            assert "file2.css" in result

    @pytest.mark.asyncio
    async def test_write_creates_directory(self):
        """Test that write_to_file creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await write_to_file(
                path="src/components/button.html",
                content="<button>Click me</button>",
                workspace_dir=tmpdir
            )
            assert "successfully" in result.lower()

            # Verify file was created
            file_path = Path(tmpdir) / "src" / "components" / "button.html"
            assert file_path.exists()


class TestConfiguration:
    """Test configuration loading."""

    def test_get_settings(self):
        """Test that settings can be loaded."""
        from app.config import get_settings
        settings = get_settings()
        assert settings is not None
        assert settings.LLM_PROVIDER in ["gemini", "openrouter"]

    def test_cors_origins_parsing(self):
        """Test CORS origins parsing."""
        from app.config import Settings

        # Test wildcard
        settings = Settings(
            LLM_PROVIDER="gemini",
            GEMINI_API_KEY="test-key",
            CORS_ORIGINS="*"
        )
        assert settings.get_cors_origins() == ["*"]

        # Test multiple origins
        settings = Settings(
            LLM_PROVIDER="gemini",
            GEMINI_API_KEY="test-key",
            CORS_ORIGINS="https://localhost:3000,https://app.example.com"
        )
        origins = settings.get_cors_origins()
        assert len(origins) == 2
        assert "https://localhost:3000" in origins


class TestAgentPrompts:
    """Test that agent personas are properly defined."""

    def test_personas_are_defined(self):
        """Test that all required personas are defined."""
        from app.graph import Personas

        assert Personas.PRODUCT_MANAGER_SYSTEM is not None
        assert len(Personas.PRODUCT_MANAGER_SYSTEM) > 0
        assert Personas.CODER_SYSTEM is not None
        assert len(Personas.CODER_SYSTEM) > 0

    def test_coder_prompt_generation(self):
        """Test coder user prompt generation."""
        from app.graph import Personas

        spec = "Create a beautiful landing page"
        prompt = Personas.get_coder_user_prompt(spec)

        assert "Build or update" in prompt
        assert spec in prompt
        assert "write_to_file" in prompt
