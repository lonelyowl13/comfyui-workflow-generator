"""
Tests for the CLI functionality.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from comfyui_workflow_generator.cli import main


class TestCLI:
    """Test cases for CLI functionality."""
    
    @pytest.fixture
    def sample_object_info(self):
        """Sample object_info for testing."""
        return {
            "CheckpointLoaderSimple": {
                "input": {
                    "required": {
                        "ckpt_name": ["STRING", {"default": "model.ckpt"}]
                    }
                },
                "output": ["MODEL", "CLIP", "VAE"]
            },
            "CLIPTextEncode": {
                "input": {
                    "required": {
                        "text": ["STRING", {"default": ""}],
                        "clip": ["CLIP", {"default": None}]
                    }
                },
                "output": ["CONDITIONING"]
            }
        }
    
    @pytest.fixture
    def temp_object_info_file(self, sample_object_info, tmp_path):
        """Create a temporary object_info.json file."""
        object_info_file = tmp_path / "object_info.json"
        with open(object_info_file, 'w') as f:
            json.dump(sample_object_info, f)
        return object_info_file
    
    @patch('sys.argv', ['comfyui-generate', 'object_info.json'])
    @patch('sys.exit')
    def test_cli_with_file(self, mock_exit, temp_object_info_file, tmp_path):
        """Test CLI with local file."""
        # Change to the temp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Mock the generator to avoid actual file operations
            with patch('comfyui_workflow_generator.cli.WorkflowGenerator') as mock_generator:
                mock_instance = MagicMock()
                mock_generator.from_file.return_value = mock_instance
                
                # Run CLI
                main()
                
                # Verify generator was called with the correct path
                mock_generator.from_file.assert_called_once()
                # Check that the call was made with the correct filename (not full path)
                call_args = mock_generator.from_file.call_args[0][0]
                assert call_args == "object_info.json"
                mock_instance.save_to_file.assert_called_once_with("workflow_api.py")
                mock_exit.assert_not_called()
                
        finally:
            os.chdir(original_cwd)
    
    @patch('sys.argv', ['comfyui-generate', 'object_info.json', '-o', 'custom_output.py'])
    @patch('sys.exit')
    def test_cli_with_custom_output(self, mock_exit, temp_object_info_file, tmp_path):
        """Test CLI with custom output file."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with patch('comfyui_workflow_generator.cli.WorkflowGenerator') as mock_generator:
                mock_instance = MagicMock()
                mock_generator.from_file.return_value = mock_instance
                
                main()
                
                mock_instance.save_to_file.assert_called_once_with("custom_output.py")
                mock_exit.assert_not_called()
                
        finally:
            os.chdir(original_cwd)
    
    @patch('sys.argv', ['comfyui-generate', 'http://127.0.0.1:8188'])
    @patch('sys.exit')
    def test_cli_with_url(self, mock_exit):
        """Test CLI with URL."""
        with patch('comfyui_workflow_generator.cli.WorkflowGenerator') as mock_generator:
            mock_instance = MagicMock()
            mock_generator.from_url.return_value = mock_instance
            
            main()
            
            mock_generator.from_url.assert_called_once_with("http://127.0.0.1:8188")
            mock_instance.save_to_file.assert_called_once_with("workflow_api.py")
            mock_exit.assert_not_called()
    
    @patch('sys.argv', ['comfyui-generate', 'nonexistent.json'])
    @patch('sys.exit')
    def test_cli_file_not_found(self, mock_exit):
        """Test CLI with non-existent file."""
        with patch('builtins.print') as mock_print:
            main()
            
            # Should exit with error code 1
            mock_exit.assert_called_with(1)
            # Verify error message was printed
            mock_print.assert_any_call("❌ Error: File not found: nonexistent.json")
    
    @patch('sys.argv', ['comfyui-generate', 'http://invalid-url'])
    @patch('sys.exit')
    def test_cli_connection_error(self, mock_exit):
        """Test CLI with connection error."""
        with patch('comfyui_workflow_generator.cli.WorkflowGenerator') as mock_generator:
            mock_generator.from_url.side_effect = ConnectionError("Connection failed")
            
            with patch('builtins.print') as mock_print:
                main()
                
                mock_exit.assert_called_once_with(1)
                mock_print.assert_any_call("❌ Error: Could not connect to server - Connection failed")
    
    @patch('sys.argv', ['comfyui-generate', 'invalid.json'])
    @patch('sys.exit')
    def test_cli_invalid_json(self, mock_exit, tmp_path):
        """Test CLI with invalid JSON file."""
        # Create a file with invalid JSON
        invalid_json_file = tmp_path / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("invalid json content")
        
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with patch('builtins.print') as mock_print:
                main()
                
                # Should exit with error code 1
                mock_exit.assert_called_with(1)
                # Verify error message was printed (check for partial match)
                error_printed = any("Invalid JSON" in str(call) for call in mock_print.call_args_list)
                assert error_printed, "Error message about invalid JSON should be printed"
                
        finally:
            os.chdir(original_cwd)
    
    @patch('sys.argv', ['comfyui-generate', 'invalid_format.json'])
    @patch('sys.exit')
    def test_cli_invalid_format(self, mock_exit, tmp_path):
        """Test CLI with invalid object_info format."""
        # Create a file with invalid format
        invalid_format_file = tmp_path / "invalid_format.json"
        with open(invalid_format_file, 'w') as f:
            json.dump({"not_object_info": "format"}, f)
        
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with patch('comfyui_workflow_generator.cli.WorkflowGenerator') as mock_generator:
                mock_generator.from_file.side_effect = KeyError("input")
                
                with patch('builtins.print') as mock_print:
                    main()
                    
                    # Should exit with error code 1
                    mock_exit.assert_called_with(1)
                    # Verify error message was printed (check for partial match)
                    error_printed = any("Invalid object_info format" in str(call) for call in mock_print.call_args_list)
                    assert error_printed, "Error message about invalid format should be printed"
                    
        finally:
            os.chdir(original_cwd) 