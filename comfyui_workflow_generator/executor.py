"""
ComfyUI Workflow Executor

Handles execution of arbitrary ComfyUI workflows and result retrieval.
"""

import os
import time
import requests
from typing import Dict, Any, Optional, List


class ComfyUIWorkflowExecutor:
    """
    Executes arbitrary ComfyUI workflows and manages results.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        """
        Initialize the ComfyUI workflow executor.
        
        Args:
            base_url: ComfyUI server URL
        """
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_server(self) -> bool:
        """Check if ComfyUI server is running."""
        try:
            response = self.session.get(f"{self.base_url}/system_stats")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def upload_image(self, image_path: str, image_name: Optional[str] = None) -> str:
        """
        Upload an image to ComfyUI.
        
        Args:
            image_path: Path to the image file to upload
            image_name: Optional custom name for the uploaded image
            
        Returns:
            Filename as stored in ComfyUI
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {"image": f}
            response = self.session.post(f"{self.base_url}/upload/image", files=files)
            
        if response.status_code != 200:
            raise Exception(f"Failed to upload image: {response.text}")
        
        result = response.json()
        uploaded_filename = result.get('name', os.path.basename(image_path))
        
        # If custom name provided, rename the file
        if image_name:
            # Note: This is a simplified approach. In practice, you might need to
            # handle file renaming on the ComfyUI server side
            uploaded_filename = image_name
        
        return uploaded_filename
    
    def upload_file(self, file_path: str, file_type: str = "image", file_name: Optional[str] = None) -> str:
        """
        Upload a file to ComfyUI.
        
        Args:
            file_path: Path to the file to upload
            file_type: Type of file ("image", "model", etc.)
            file_name: Optional custom name for the uploaded file
            
        Returns:
            Filename as stored in ComfyUI
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {file_type: f}
            response = self.session.post(f"{self.base_url}/upload/{file_type}", files=files)
            
        if response.status_code != 200:
            raise Exception(f"Failed to upload file: {response.text}")
        
        result = response.json()
        uploaded_filename = result.get('name', os.path.basename(file_path))
        
        # If custom name provided, rename the file
        if file_name:
            # Note: This is a simplified approach. In practice, you might need to
            # handle file renaming on the ComfyUI server side
            uploaded_filename = file_name
        
        return uploaded_filename
    
    def queue_workflow(self, workflow: Dict[str, Any]) -> str:
        """
        Queue a workflow for execution.
        
        Args:
            workflow: Workflow dictionary
            
        Returns:
            Prompt ID for tracking execution
        """
        payload = {"prompt": workflow}
        response = self.session.post(f"{self.base_url}/prompt", json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Failed to queue workflow: {response.text}")
        
        result = response.json()
        return result['prompt_id']
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Wait for workflow completion.
        
        Args:
            prompt_id: Prompt ID from queue_workflow
            timeout: Timeout in seconds
            
        Returns:
            Workflow execution result
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.session.get(f"{self.base_url}/history/{prompt_id}")
            
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    return history[prompt_id]
            
            time.sleep(1)
        
        raise TimeoutError(f"Workflow execution timed out after {timeout} seconds")
    
    def download_results(self, output_data: Dict[str, Any], output_dir: str = ".") -> List[str]:
        """
        Download all results from workflow execution.
        
        Args:
            output_data: Workflow execution result
            output_dir: Directory to save results
            
        Returns:
            List of downloaded file paths
        """
        if not output_data.get('outputs'):
            raise Exception("No outputs found in workflow result")
        
        downloaded_files = []

        os.makedirs(output_dir, exist_ok=True)
        
        for node_id, node_output in output_data['outputs'].items():
            if 'images' in node_output:
                for image_info in node_output['images']:
                    filename = image_info['filename']
                    
                    # Download the image
                    response = self.session.get(f"{self.base_url}/view?filename={filename}")
                    
                    if response.status_code != 200:
                        print(f"Warning: Failed to download {filename}")
                        continue
                    
                    # Save to output directory
                    output_path = os.path.join(output_dir, filename)
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    
                    downloaded_files.append(output_path)
        
        return downloaded_files
    
    def execute_workflow(self, 
                        workflow: Dict[str, Any], 
                        output_dir: str = ".", 
                        timeout: int = 300) -> List[str]:
        """
        Execute a workflow dictionary as-is without any modifications.
        
        Args:
            workflow: Workflow dictionary (should be complete with all inputs)
            output_dir: Directory to save results
            timeout: Execution timeout in seconds
            
        Returns:
            List of result file paths
        """
        # Check if server is running
        if not self.check_server():
            raise Exception("ComfyUI server is not running. Please start ComfyUI first.")
        
        # Queue workflow (no modifications)
        prompt_id = self.queue_workflow(workflow)
        
        # Wait for completion
        result = self.wait_for_completion(prompt_id, timeout)
        
        # Download results
        result_files = self.download_results(result, output_dir)
        
        return result_files 