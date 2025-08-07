"""
ComfyUI Workflow Generator

A Python package for generating ComfyUI workflow APIs from object_info.json
"""

from .generator import WorkflowGenerator
from .executor import ComfyUIWorkflowExecutor

__version__ = "0.1.0"
__all__ = [
    "WorkflowGenerator",
    "ComfyUIWorkflowExecutor"
] 