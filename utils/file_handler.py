import os
import json
import ast
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from config.settings import settings

class FileHandler:
    @staticmethod
    def read_file(file_path: Path) -> Optional[str]:
        """Read file content safely"""
        try:
            if file_path.stat().st_size > settings.MAX_FILE_SIZE:
                print(f"File {file_path} is too large (>{settings.MAX_FILE_SIZE} bytes)")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    @staticmethod
    def write_file(file_path: Path, content: str) -> bool:
        """Write content to file safely"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error writing file {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: Path) -> Optional[Dict]:
        """Load JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {file_path}: {e}")
            return None
    
    @staticmethod
    def save_json(file_path: Path, data: Dict) -> bool:
        """Save data to JSON file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving JSON to {file_path}: {e}")
            return False
    
    @staticmethod
    def find_files(directory: Path, extensions: List[str] = None) -> List[Path]:
        """Find files with specific extensions in directory"""
        if extensions is None:
            extensions = settings.SUPPORTED_EXTENSIONS
        
        files = []
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix in extensions:
                    files.append(file_path)
        except Exception as e:
            print(f"Error scanning directory {directory}: {e}")
        
        return files
    
    @staticmethod
    def parse_python_file(file_path: Path) -> Optional[Dict]:
        """Parse Python file and extract structure"""
        content = FileHandler.read_file(file_path)
        if not content:
            return None
        
        try:
            tree = ast.parse(content)
            structure = {
                'functions': [],
                'classes': [],
                'imports': [],
                'variables': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    structure['functions'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args]
                    })
                elif isinstance(node, ast.ClassDef):
                    structure['classes'].append({
                        'name': node.name,
                        'line': node.lineno
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            structure['imports'].append(alias.name)
                    else:
                        module = node.module or ''
                        for alias in node.names:
                            structure['imports'].append(f"{module}.{alias.name}")
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            structure['variables'].append(target.id)
            
            return structure
        except Exception as e:
            print(f"Error parsing Python file {file_path}: {e}")
            return None
    
    @staticmethod
    def backup_file(file_path: Path) -> bool:
        """Create backup of file"""
        try:
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            shutil.copy2(file_path, backup_path)
            return True
        except Exception as e:
            print(f"Error creating backup for {file_path}: {e}")
            return False
    
    @staticmethod
    def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python code syntax"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    @staticmethod
    def get_file_info(file_path: Path) -> Dict:
        """Get comprehensive file information"""
        try:
            stat = file_path.stat()
            return {
                'path': str(file_path),
                'name': file_path.name,
                'extension': file_path.suffix,
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'is_python': file_path.suffix == '.py'
            }
        except Exception as e:
            print(f"Error getting file info for {file_path}: {e}")
            return {}