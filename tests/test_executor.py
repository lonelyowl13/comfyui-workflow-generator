"""
Tests for the ComfyUIWorkflowExecutor class.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from comfyui_workflow_generator.executor import ComfyUIWorkflowExecutor


class TestComfyUIWorkflowExecutor:
    """Test cases for ComfyUIWorkflowExecutor."""
    
    @pytest.fixture
    def executor(self):
        """Create a ComfyUIWorkflowExecutor instance."""
        return ComfyUIWorkflowExecutor("http://127.0.0.1:8188")
    
    @pytest.fixture
    def sample_workflow(self):
        """Sample workflow for testing."""
        return {
            "1": {
                "inputs": {
                    "ckpt_name": "v1-5-pruned.ckpt"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": "Hello, world!",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            }
        }
    
    def test_executor_initialization(self, executor: ComfyUIWorkflowExecutor):
        """Test executor initialization."""
        assert executor.base_url == "http://127.0.0.1:8188"
        assert executor.session is not None
    
    @patch('requests.Session.get')
    def test_check_server_success(self, mock_get, executor: ComfyUIWorkflowExecutor):
        """Test server check when server is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        assert executor.check_server() is True
        mock_get.assert_called_once_with("http://127.0.0.1:8188/system_stats")
    
    @patch('requests.Session.get')
    def test_check_server_failure(self, mock_get, executor: ComfyUIWorkflowExecutor):
        """Test server check when server is not running."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        # The method should catch the exception and return False
        result = executor.check_server()
        assert result is False
    
    @patch('requests.Session.post')
    def test_queue_workflow_success(self, mock_post, executor: ComfyUIWorkflowExecutor, sample_workflow: Dict[str, Any]):
        """Test successful workflow queuing."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prompt_id": "test_prompt_123"}
        mock_post.return_value = mock_response
        
        prompt_id = executor.queue_workflow(sample_workflow)
        
        assert prompt_id == "test_prompt_123"
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8188/prompt",
            json={"prompt": sample_workflow}
        )
    
    @patch('requests.Session.post')
    def test_queue_workflow_failure(self, mock_post, executor, sample_workflow):
        """Test workflow queuing failure."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response
        
        with pytest.raises(Exception, match="Failed to queue workflow: Internal server error"):
            executor.queue_workflow(sample_workflow)
    
    @patch('requests.Session.get')
    def test_wait_for_completion_success(self, mock_get, executor):
        """Test successful workflow completion waiting."""
        # Mock the history response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "test_prompt_123": {
                "outputs": {
                    "1": {"images": [{"filename": "test.png"}]}
                }
            }
        }
        mock_get.return_value = mock_response
        
        result = executor.wait_for_completion("test_prompt_123", timeout=5)
        
        assert result["outputs"]["1"]["images"][0]["filename"] == "test.png"
        mock_get.assert_called_with("http://127.0.0.1:8188/history/test_prompt_123")
    
    @patch('requests.Session.get')
    def test_wait_for_completion_timeout(self, mock_get, executor):
        """Test workflow completion timeout."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # No prompt_id in history
        mock_get.return_value = mock_response
        
        with pytest.raises(TimeoutError, match="Workflow execution timed out after 1 seconds"):
            executor.wait_for_completion("test_prompt_123", timeout=1)
    
    @patch('requests.Session.get')
    def test_download_results_success(self, mock_get, executor, tmp_path):
        """Test successful result downloading."""
        # Mock the image download response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response
        
        output_data = {
            "outputs": {
                "1": {
                    "images": [
                        {"filename": "test_image.png"}
                    ]
                }
            }
        }
        
        result_files = executor.download_results(output_data, str(tmp_path))
        
        assert len(result_files) == 1
        assert result_files[0].endswith("test_image.png")
        assert os.path.exists(result_files[0])
        
        # Verify the image content
        with open(result_files[0], 'rb') as f:
            assert f.read() == b"fake_image_data"
    
    @patch('requests.Session.get')
    def test_download_results_no_outputs(self, mock_get, executor):
        """Test downloading when no outputs exist."""
        output_data = {"outputs": {}}
        
        with pytest.raises(Exception, match="No outputs found in workflow result"):
            executor.download_results(output_data)
    
    @patch('requests.Session.get')
    def test_download_results_download_failure(self, mock_get, executor, tmp_path):
        """Test result downloading when image download fails."""
        # Mock the image download failure
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        output_data = {
            "outputs": {
                "1": {
                    "images": [
                        {"filename": "missing_image.png"}
                    ]
                }
            }
        }
        
        # Should not raise exception, just skip the failed download
        result_files = executor.download_results(output_data, str(tmp_path))
        
        assert len(result_files) == 0
    
    @patch.object(ComfyUIWorkflowExecutor, 'check_server')
    @patch.object(ComfyUIWorkflowExecutor, 'queue_workflow')
    @patch.object(ComfyUIWorkflowExecutor, 'wait_for_completion')
    @patch.object(ComfyUIWorkflowExecutor, 'download_results')
    def test_execute_workflow_success(self, mock_download, mock_wait, mock_queue, mock_check, executor, sample_workflow, tmp_path):
        """Test successful workflow execution."""
        mock_check.return_value = True
        mock_queue.return_value = "test_prompt_123"
        mock_wait.return_value = {"outputs": {"1": {"images": [{"filename": "test.png"}]}}}
        mock_download.return_value = [str(tmp_path / "test.png")]
        
        result_files = executor.execute_workflow(sample_workflow, str(tmp_path))
        
        assert result_files == [str(tmp_path / "test.png")]
        mock_check.assert_called_once()
        mock_queue.assert_called_once_with(sample_workflow)
        mock_wait.assert_called_once_with("test_prompt_123", 300)
        mock_download.assert_called_once()
    
    @patch.object(ComfyUIWorkflowExecutor, 'check_server')
    def test_execute_workflow_server_not_running(self, mock_check, executor, sample_workflow):
        """Test workflow execution when server is not running."""
        mock_check.return_value = False
        
        with pytest.raises(Exception, match="ComfyUI server is not running"):
            executor.execute_workflow(sample_workflow)
    
    @patch('requests.Session.post')
    def test_upload_image_success(self, mock_post, executor, tmp_path):
        """Test successful image upload."""
        # Create a temporary image file
        image_file = tmp_path / "test_image.png"
        with open(image_file, 'wb') as f:
            f.write(b"fake_image_data")
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "uploaded_image.png"}
        mock_post.return_value = mock_response
        
        filename = executor.upload_image(str(image_file))
        
        assert filename == "uploaded_image.png"
        mock_post.assert_called_once()
    
    @patch('requests.Session.post')
    def test_upload_image_with_custom_name(self, mock_post, executor, tmp_path):
        """Test image upload with custom name."""
        image_file = tmp_path / "test_image.png"
        with open(image_file, 'wb') as f:
            f.write(b"fake_image_data")
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "uploaded_image.png"}
        mock_post.return_value = mock_response
        
        filename = executor.upload_image(str(image_file), "custom_name.png")
        
        assert filename == "custom_name.png"
    
    def test_upload_image_file_not_found(self, executor):
        """Test image upload with non-existent file."""
        with pytest.raises(FileNotFoundError):
            executor.upload_image("nonexistent.png")
    
    @patch('requests.Session.post')
    def test_upload_file_success(self, mock_post, executor, tmp_path):
        """Test successful file upload."""
        file_path = tmp_path / "test_file.txt"
        with open(file_path, 'w') as f:
            f.write("test content")
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "uploaded_file.txt"}
        mock_post.return_value = mock_response
        
        filename = executor.upload_file(str(file_path), "text")
        
        assert filename == "uploaded_file.txt"
        # Verify the post was called with the correct URL and files
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://127.0.0.1:8188/upload/text"
        assert "text" in call_args[1]["files"]
    
    def test_upload_file_not_found(self, executor):
        """Test file upload with non-existent file."""
        with pytest.raises(FileNotFoundError):
            executor.upload_file("nonexistent.txt") 