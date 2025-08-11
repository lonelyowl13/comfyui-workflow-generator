"""
Integration tests for the complete workflow generation and execution process.
"""

import json
import tempfile
import os
import importlib.util
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from comfyui_workflow_generator.generator import WorkflowGenerator
from comfyui_workflow_generator.executor import ComfyUIWorkflowExecutor


class TestIntegration:
    """Integration tests for the complete workflow system."""
    
    @pytest.fixture
    def sample_object_info(self):
        """Sample object_info for integration testing."""
        return {
            "CheckpointLoaderSimple": {
                "input": {
                    "required": {
                        "ckpt_name": ["STRING", {"default": "v1-5-pruned.ckpt"}]
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
            },
            "KSampler": {
                "input": {
                    "required": {
                        "seed": ["INT", {"default": 0}],
                        "steps": ["INT", {"default": 20}],
                        "cfg": ["FLOAT", {"default": 8.0}],
                        "sampler_name": ["STRING", {"default": "euler"}],
                        "scheduler": ["STRING", {"default": "normal"}],
                        "denoise": ["FLOAT", {"default": 1.0}],
                        "model": ["MODEL", {"default": None}],
                        "positive": ["CONDITIONING", {"default": None}],
                        "negative": ["CONDITIONING", {"default": None}],
                        "latent_image": ["LATENT", {"default": None}]
                    },
                    "optional": {
                        "add_noise": ["BOOLEAN", {"default": True}],
                        "return_with_leftover_noise": ["BOOLEAN", {"default": False}]
                    }
                },
                "output": ["LATENT"]
            },
            "VAELoader": {
                "input": {
                    "required": {
                        "vae_name": ["STRING", {"default": "sdxl_vae.safetensors"}]
                    }
                },
                "output": ["VAE"]
            },
            "LoadImage": {
                "input": {
                    "required": {
                        "image": ["STRING", {"default": ""}]
                    }
                },
                "output": ["IMAGE", "MASK"]
            },
            "VAEEncode": {
                "input": {
                    "required": {
                        "pixels": ["IMAGE", {"default": None}],
                        "vae": ["VAE", {"default": None}]
                    }
                },
                "output": ["LATENT"]
            },
            "VAEDecode": {
                "input": {
                    "required": {
                        "samples": ["LATENT", {"default": None}],
                        "vae": ["VAE", {"default": None}]
                    }
                },
                "output": ["IMAGE"]
            },
            "SaveImage": {
                "input": {
                    "required": {
                        "images": ["IMAGE", {"default": None}],
                        "filename_prefix": ["STRING", {"default": "ComfyUI"}]
                    }
                },
                "output": []
            }
        }
    
    def test_complete_workflow_generation(self, sample_object_info, tmp_path):
        """Test complete workflow generation from object_info."""
        # Generate workflow API
        generator = WorkflowGenerator(sample_object_info)
        output_file = tmp_path / "workflow_api.py"
        generator.save_to_file(str(output_file))
        
        # Verify the generated file
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        
        # Read and verify the generated code
        with open(output_file, 'r') as f:
            code = f.read()
            
            # Should contain all expected classes
            assert "class Workflow:" in code
            assert "class NodeOutput:" in code
            assert "class BoolNodeOutput(" in code  # Note: no colon in actual output
            
            # Should contain all expected methods
            assert "def CheckpointLoaderSimple(" in code
            assert "def CLIPTextEncode(" in code
            assert "def KSampler(" in code
            assert "def VAELoader(" in code
            assert "def LoadImage(" in code
            assert "def VAEEncode(" in code
            assert "def VAEDecode(" in code
            assert "def SaveImage(" in code
            
            # Should contain utility functions
            assert "def to_comfy_input(" in code
            assert "def random_node_id(" in code
    
    def test_generated_workflow_structure(self, sample_object_info):
        """Test that generated workflow has correct structure."""
        generator = WorkflowGenerator(sample_object_info)
        workflow_class = generator.generate_workflow_class()
        
        # Get all method names
        method_names = [method.name for method in workflow_class.body if hasattr(method, 'name')]
        
        # Should have all node methods
        expected_methods = [
            "CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", 
            "VAELoader", "LoadImage", "VAEEncode", "VAEDecode", "SaveImage"
        ]
        for method in expected_methods:
            assert method in method_names
        
        # Should have required infrastructure methods
        assert "__init__" in method_names
        assert "_add_node" in method_names
        assert "get_workflow" in method_names

    
    def test_custom_type_generation_integration(self, sample_object_info):
        """Test that custom types are generated correctly."""
        generator = WorkflowGenerator(sample_object_info)
        
        # Generate custom types
        custom_types = generator.generate_custom_types()
        type_names = [cls.name for cls in custom_types]

        # Should include all custom types from the object_info
        expected_types = ["MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK"]
        for type_name in expected_types:
            assert type_name in type_names
        
        # Should not include primitive types
        primitive_types = ["int", "float", "str", "bool"]
        for type_name in primitive_types:
            assert type_name not in type_names
    
    def test_workflow_execution_integration(self, sample_object_info, tmp_path):
        """Test complete workflow execution integration."""
        # Generate workflow API
        generator = WorkflowGenerator(sample_object_info)
        output_file = tmp_path / "workflow_api.py"
        generator.save_to_file(str(output_file))

        # Import the generated workflow API
        spec = importlib.util.spec_from_file_location("workflow_module", str(output_file))
        workflow_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow_module)
        
        # Create workflow using the generated API
        wf = workflow_module.Workflow()
        
        # Use the generated methods to create a real workflow
        model, clip, vae = wf.CheckpointLoaderSimple(ckpt_name="v1-5-pruned.ckpt")
        positive = wf.CLIPTextEncode(text="Hello, world!", clip=clip)
        
        # Get the workflow dictionary from the generated API
        workflow = wf.workflow_dict
        
        # Test executor with the workflow
        executor = ComfyUIWorkflowExecutor("http://127.0.0.1:8188")
        
        # Mock the executor methods to avoid actual server calls
        with patch.object(executor, 'check_server', return_value=True), \
             patch.object(executor, 'queue_workflow', return_value="test_prompt_123"), \
             patch.object(executor, 'wait_for_completion') as mock_wait, \
             patch.object(executor, 'download_results') as mock_download:
            
            mock_wait.return_value = {
                "outputs": {
                    "1": {"images": [{"filename": "test.png"}]}
                }
            }
            mock_download.return_value = [str(tmp_path / "test.png")]
            
            # Execute workflow
            result_files = executor.execute_workflow(workflow, str(tmp_path))
            
            # Verify execution
            assert result_files == [str(tmp_path / "test.png")]
            executor.check_server.assert_called_once()
            executor.queue_workflow.assert_called_once_with(workflow)
            mock_wait.assert_called_once_with("test_prompt_123", 300)
            mock_download.assert_called_once()
    
    def test_file_operations_integration(self, sample_object_info, tmp_path):
        """Test file operations integration."""
        # Test from_file method
        object_info_file = tmp_path / "object_info.json"
        with open(object_info_file, 'w') as f:
            json.dump(sample_object_info, f)
        
        # Create generator from file
        generator = WorkflowGenerator.from_file(str(object_info_file))
        assert generator.object_info == sample_object_info
        
        # Test save_to_file method
        output_file = tmp_path / "generated_api.py"
        generator.save_to_file(str(output_file))
        
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        
        # Verify file content
        with open(output_file, 'r') as f:
            content = f.read()
            assert "class Workflow:" in content
            assert "class NodeOutput:" in content
    
    def test_error_handling_integration(self, sample_object_info):
        """Test error handling in the complete workflow."""
        generator = WorkflowGenerator(sample_object_info)
        
        # Test with invalid node info
        invalid_node_info = {
            "InvalidNode": {
                "input": {},  # No inputs
                "output": []
            }
        }
        
        # Should handle nodes with no inputs gracefully
        method = generator.generate_node_method("InvalidNode", invalid_node_info["InvalidNode"])
        assert method.name == "InvalidNode"
        assert len(method.args.args) == 1  # Only self parameter
        
        # Test with empty object_info
        empty_generator = WorkflowGenerator({})
        custom_types = empty_generator.generate_custom_types()
        assert len(custom_types) == 0
        
        workflow_class = empty_generator.generate_workflow_class()
        assert workflow_class.name == "Workflow"
        
        # Should still have required methods
        method_names = [method.name for method in workflow_class.body if hasattr(method, 'name')]
        assert "__init__" in method_names
        assert "_add_node" in method_names
        assert "get_workflow" in method_names 