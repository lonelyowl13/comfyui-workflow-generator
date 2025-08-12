"""
Tests for the WorkflowGenerator class.
"""

import json
import ast
import pytest

from comfyui_workflow_generator.generator import WorkflowGenerator


class TestWorkflowGenerator:
    """Test cases for WorkflowGenerator."""
    
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
                        "vae_name": ["STRING", {"default": "vae.pt"}]
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
            }
        }
    
    def test_generator_initialization(self, sample_object_info):
        """Test that WorkflowGenerator initializes correctly."""
        generator = WorkflowGenerator(sample_object_info)
        assert generator.object_info == sample_object_info
        assert "BOOLEAN" in generator.primitives
        assert generator.primitives["BOOLEAN"] == "bool"
    
    def test_normalize_name(self, sample_object_info):
        """Test name normalization."""
        generator = WorkflowGenerator(sample_object_info)
        
        # Test basic normalization
        assert generator.normalize_name("CheckpointLoaderSimple") == "CheckpointLoaderSimple"
        assert generator.normalize_name("CLIP Text Encode") == "CLIP_Text_Encode"
        assert generator.normalize_name("KSampler Advanced") == "KSampler_Advanced"
        
        # Test special characters
        assert generator.normalize_name("Node@#$%😭😭😭") == "Node"
        assert generator.normalize_name("123Node") == "_123Node"
        
        # Test Python keywords
        assert generator.normalize_name("class") == "class_"
        assert generator.normalize_name("def") == "def_"
    
    def test_get_normalized_type(self, sample_object_info):
        """Test type normalization."""
        generator = WorkflowGenerator(sample_object_info)
        
        # Test primitive types
        assert generator.get_normalized_type("INT") == "int"
        assert generator.get_normalized_type("FLOAT") == "float"
        assert generator.get_normalized_type("STRING") == "str"
        assert generator.get_normalized_type("BOOLEAN") == "bool"
        
        # Test custom types
        assert generator.get_normalized_type("MODEL") == "MODEL"
        assert generator.get_normalized_type("CLIP") == "CLIP"
        assert generator.get_normalized_type("VAE") == "VAE"

        # comfyui isn't super strict about node names and return types, yeah
        assert generator.get_normalized_type("a, bunch, of, random, stupid, shit") == "a_bunch_of_random_stupid_shit"
        
        # Test wildcard type
        assert generator.get_normalized_type("*") == "AnyNodeOutput"
        
        # Test list (enum)
        assert generator.get_normalized_type(["a", "b", "c"]) == "str"
        
        # Test with node_name and input_name for enum mapping (should return primitive types)
        assert generator.get_normalized_type(["a", "b", "c"], "TestNode", "test_input") == "str"
        assert generator.get_normalized_type([1, 2, 3], "TestNode", "test_input") == "int"

    def test_get_return_type(self, sample_object_info):
        """Test return type generation."""
        generator = WorkflowGenerator(sample_object_info)
        
        # Test single output
        single_output = generator.get_return_type(["MODEL"])
        assert isinstance(single_output, type(ast.Name(id="MODEL", ctx=ast.Load())))
        
        # Test multiple outputs
        multiple_outputs = generator.get_return_type(["MODEL", "CLIP", "VAE"])
        assert isinstance(multiple_outputs, type(ast.Tuple(elts=[], ctx=ast.Load())))
        
        # Test primitive types
        int_output = generator.get_return_type(["INT"])
        assert isinstance(int_output, type(ast.Name(id="IntNodeOutput", ctx=ast.Load())))
        
        bool_output = generator.get_return_type(["BOOLEAN"])
        assert isinstance(bool_output, type(ast.Name(id="BoolNodeOutput", ctx=ast.Load())))
    
    def test_generate_custom_types(self, sample_object_info):
        """Test custom type generation."""
        generator = WorkflowGenerator(sample_object_info)
        custom_types = generator.generate_custom_types()
        
        # Should generate classes for custom types
        type_names = [cls.name for cls in custom_types]
        assert "MODEL" in type_names
        assert "CLIP" in type_names
        assert "VAE" in type_names
        assert "CONDITIONING" in type_names
        assert "LATENT" in type_names
        assert "IMAGE" in type_names
        assert "MASK" in type_names
        
        # Should not include primitive types
        assert "int" not in type_names
        assert "float" not in type_names
        assert "str" not in type_names
        assert "bool" not in type_names
    
    def test_int_enum_generation(self):
        """Test integer enum generation."""
        object_info = {
            'TestNode': {
                'input': {
                    'required': {
                        'duration': [[5, 8, 10], 'COMBO']
                    }
                },
                'output': ['LATENT']
            }
        }
        generator = WorkflowGenerator(object_info)
        custom_types = generator.generate_custom_types()
        
        # Should generate an IntEnum
        enum_names = [cls.name for cls in custom_types if hasattr(cls, 'name')]
        assert "TestNodedurationValues" in enum_names
        
        # Check that it's an IntEnum
        for cls in custom_types:
            if cls.name == "TestNodedurationValues":
                assert "IntEnum" in [base.id for base in cls.bases]
                break
    
    def test_empty_enum_generation(self):
        """Test empty enum generation."""
        object_info = {
            'TestNode': {
                'input': {
                    'required': {
                        'empty_enum': [[], 'COMBO']  # Empty enum
                    }
                },
                'output': ['LATENT']
            }
        }
        generator = WorkflowGenerator(object_info)
        custom_types = generator.generate_custom_types()
        
        # Should generate an enum even if empty
        enum_names = [cls.name for cls in custom_types if hasattr(cls, 'name')]
        assert "TestNodeempty_enumValues" in enum_names
        
        # Check that it has a pass statement
        for cls in custom_types:
            if cls.name == "TestNodeempty_enumValues":
                # Should have at least one body element (the pass statement)
                assert len(cls.body) > 0
                # Check if it has a pass statement
                has_pass = any(isinstance(stmt, ast.Pass) for stmt in cls.body)
                assert has_pass, "Empty enum should have a pass statement"
                break
    
    def test_generate_base_classes(self, sample_object_info):
        """Test base class generation."""
        generator = WorkflowGenerator(sample_object_info)
        base_classes = generator.generate_base_classes()
        
        class_names = [cls.name for cls in base_classes]
        assert "NodeOutput" in class_names
        assert "StrNodeOutput" in class_names
        assert "FloatNodeOutput" in class_names
        assert "IntNodeOutput" in class_names
        assert "BoolNodeOutput" in class_names
        assert "AnyNodeOutput" in class_names
    
    def test_generate_node_method(self, sample_object_info):
        """Test node method generation."""
        generator = WorkflowGenerator(sample_object_info)
        
        # Test CheckpointLoaderSimple method generation
        method = generator.generate_node_method("CheckpointLoaderSimple", sample_object_info["CheckpointLoaderSimple"])
        assert method.name == "CheckpointLoaderSimple"
        assert len(method.args.args) == 2  # self + ckpt_name
        
        # Test KSampler method generation (with boolean inputs)
        ksampler_method = generator.generate_node_method("KSampler", sample_object_info["KSampler"])
        assert ksampler_method.name == "KSampler"

        # Should have all required and optional inputs
        expected_args = ["self", "seed", "steps", "cfg", "sampler_name", "scheduler", 
                        "denoise", "model", "positive", "negative", "latent_image",
                        "add_noise", "return_with_leftover_noise"]
        actual_args = [arg.arg for arg in ksampler_method.args.args]
        assert actual_args == expected_args
    
    def test_generate_workflow_class(self, sample_object_info):
        """Test workflow class generation."""
        generator = WorkflowGenerator(sample_object_info)
        workflow_class = generator.generate_workflow_class()
        
        assert workflow_class.name == "Workflow"
        
        # Should have methods for each node
        method_names = [method.name for method in workflow_class.body if hasattr(method, 'name')]
        assert "CheckpointLoaderSimple" in method_names
        assert "CLIPTextEncode" in method_names
        assert "KSampler" in method_names
        assert "VAELoader" in method_names
        assert "LoadImage" in method_names
        
        # Should have required methods
        assert "__init__" in method_names
        assert "_add_node" in method_names
        assert "get_workflow" in method_names
    
    def test_generate_code(self, sample_object_info):
        """Test code generation."""
        generator = WorkflowGenerator(sample_object_info)
        code = generator.genetate_module_code()
        
        # Should be valid Python code
        assert isinstance(code, str)
        assert len(code) > 0
        
        # Should contain expected classes
        assert "class Workflow:" in code
        assert "class NodeOutput:" in code
        assert "class BoolNodeOutput(" in code
        
        # Should contain expected methods
        assert "def CheckpointLoaderSimple(" in code
        assert "def KSampler(" in code
        
        # Should contain boolean handling
        assert "bool" in code
        assert "BoolNodeOutput" in code
    
    def test_save_to_file(self, sample_object_info, tmp_path):
        """Test saving generated code to file."""
        generator = WorkflowGenerator(sample_object_info)
        output_file = tmp_path / "test_workflow_api.py"
        
        generator.save_to_file(str(output_file))
        
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        
        # Verify the file contains valid Python code
        with open(output_file, 'r') as f:
            content = f.read()
            assert "class Workflow:" in content
            assert "class NodeOutput:" in content
    
    def test_from_file(self, sample_object_info, tmp_path):
        """Test creating generator from file."""
        # Create temporary object_info file
        object_info_file = tmp_path / "object_info.json"
        with open(object_info_file, 'w') as f:
            json.dump(sample_object_info, f)
        
        # Create generator from file
        generator = WorkflowGenerator.from_file(str(object_info_file))
        
        assert generator.object_info == sample_object_info
        assert "CheckpointLoaderSimple" in generator.object_info
    
    def test_generate_module(self, sample_object_info):
        """Test complete module generation."""
        generator = WorkflowGenerator(sample_object_info)
        module = generator.generate_module()
        
        # Should have imports
        import_nodes = [node for node in module.body if hasattr(node, 'module')]
        assert len(import_nodes) > 0
        
        # Should have classes
        class_nodes = [node for node in module.body if hasattr(node, 'name') and hasattr(node, 'bases')]
        assert len(class_nodes) > 0
        
        # Should have functions
        function_nodes = [node for node in module.body if hasattr(node, 'name') and hasattr(node, 'args')]
        assert len(function_nodes) > 0
    
    def test_empty_object_info(self):
        """Test handling of empty object_info."""
        generator = WorkflowGenerator({})
        
        # Should handle empty object_info gracefully
        custom_types = generator.generate_custom_types()
        assert len(custom_types) == 0
        
        workflow_class = generator.generate_workflow_class()
        assert workflow_class.name == "Workflow"
        
        # Should still have required methods
        method_names = [method.name for method in workflow_class.body if hasattr(method, 'name')]
        assert "__init__" in method_names
        assert "_add_node" in method_names
        assert "get_workflow" in method_names 