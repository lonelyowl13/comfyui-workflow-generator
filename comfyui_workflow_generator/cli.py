#!/usr/bin/env python3
"""
Command-line interface for ComfyUI Workflow Generator
"""

import argparse
import sys
import json
import requests
from pathlib import Path
from .generator import WorkflowGenerator


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate ComfyUI workflow API from object_info.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from local file
  comfyui-generate object_info.json -o my_api.py
  
  # Generate from ComfyUI server
  comfyui-generate http://127.0.0.1:8188 -o workflow_api.py
  
  # Generate with default output name
  comfyui-generate object_info.json
        """
    )
    
    parser.add_argument(
        "source",
        help="Source of object_info: file path or URL (e.g., object_info.json or http://127.0.0.1:8188)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="workflow_api.py",
        help="Output file path (default: workflow_api.py)"
    )
    
    args = parser.parse_args()
    
    try:
        # Auto-detect if source is URL or file
        if args.source.startswith(('http://', 'https://')):
            print(f"🔗 Generating API from URL: {args.source}")
            generator = WorkflowGenerator.from_url(args.source)
        else:
            # Check if file exists
            file_path = Path(args.source)
            if not file_path.exists():
                print(f"❌ Error: File not found: {args.source}")
                print(f"💡 Tip: Make sure the file exists and the path is correct")
                sys.exit(1)
            
            print(f"📁 Generating API from file: {args.source}")
            generator = WorkflowGenerator.from_file(args.source)
        
        # Generate and save
        generator.save_to_file(args.output)
        print(f"✅ Generated workflow API: {args.output}")
        print(f"💡 You can now import and use it:")
        print(f"   from {Path(args.output).stem} import Workflow")
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        sys.exit(1)
    except ConnectionError as e:
        print(f"❌ Error: Could not connect to server - {e}")
        print(f"💡 Tip: Make sure the ComfyUI server is running and accessible")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Network request failed - {e}")
        print(f"💡 Tip: Check your internet connection and server URL")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in object_info - {e}")
        print(f"💡 Tip: Make sure the file contains valid JSON")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Error: Invalid object_info format - missing key: {e}")
        print(f"💡 Tip: Make sure the file contains valid ComfyUI object_info")
        print(f"   Expected format: {{'NodeName': {{'input': {{'required': {{...}}}}}}}}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"💡 Tip: Check that the source contains valid ComfyUI object_info")
        sys.exit(1)


if __name__ == "__main__":
    main() 