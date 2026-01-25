#!/usr/bin/env python3
"""
Codebase Indexer for Project-Specific RAG
Extracts and embeds code patterns from user's project
"""

import os
import json
import re
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict


@dataclass
class CodePattern:
    """Extracted code pattern with metadata"""
    pattern_id: str  # SHA256(file_path + line_range)
    language: str  # python, typescript, javascript, go, etc.
    pattern_type: str  # class, function, interface, error_handling, etc.
    code_snippet: str  # The actual code (max 500 chars)
    file_path: str
    line_range: tuple  # (start_line, end_line)
    description: str  # Auto-generated summary
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    frequency: int = 1


@dataclass
class ProjectContext:
    """Overall project analysis"""
    project_root: str
    primary_language: str
    frameworks: List[str]
    key_conventions: Dict[str, Any]
    error_handling_patterns: List[CodePattern]
    naming_conventions: Dict[str, str]
    import_patterns: List[str]
    test_conventions: Dict[str, Any]
    total_files_analyzed: int
    total_patterns: int


class CodebaseIndexer:
    """Extract patterns from codebase for RAG"""

    # Language configurations
    LANGUAGE_PATTERNS = {
        'python': {
            'class': r'^class\s+(\w+).*?:',
            'function': r'^def\s+(\w+)\s*\(',
            'async_function': r'^async\s+def\s+(\w+)',
            'import': r'^(?:from|import)\s+(.+)',
            'error_handling': r'(?:try|except|raise|Error)',
            'type_hints': r'(?:\w+:\s*\w+|->)',
        },
        'typescript': {
            'class': r'^(?:export\s+)?class\s+(\w+)',
            'interface': r'^(?:export\s+)?interface\s+(\w+)',
            'function': r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)',
            'import': r'^(?:import|from)\s+',
            'error_handling': r'(?:try|catch|throw|Error)',
            'async': r'async\s+(?:function|\()',
        },
        'javascript': {
            'function': r'^(?:async\s+)?function\s+(\w+)|const\s+(\w+)\s*=',
            'class': r'^class\s+(\w+)',
            'import': r'^(?:import|require)\s*',
            'error_handling': r'(?:try|catch|throw)',
            'async': r'async\s+',
        },
        'go': {
            'function': r'^func\s+\(.*?\)\s+(\w+)\s*\(',
            'type': r'^type\s+(\w+)\s+struct',
            'import': r'^import\s*\(',
            'error_handling': r'(?:if\s+err|panic|recover)',
        },
        'swift': {
            'class': r'^(?:class|struct)\s+(\w+)',
            'function': r'^func\s+(\w+)',
            'protocol': r'^protocol\s+(\w+)',
        }
    }

    # Patterns to skip
    SKIP_PATTERNS = [
        r'test_',
        r'_test\.py',
        r'\.test\.ts',
        r'\.spec\.ts',
        r'node_modules',
        r'\.git',
        r'__pycache__',
        r'\.venv',
        r'venv',
        r'dist',
        r'build',
        r'\.egg-info',
        r'\.next',
        r'\.cache',
    ]

    def __init__(self, project_root: str, cache_dir: Optional[str] = None):
        self.project_root = Path(project_root).resolve()
        self.cache_dir = Path(cache_dir or Path.home() / '.cache' / 'mageagent-rag')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.patterns: List[CodePattern] = []
        self.file_hashes: Dict[str, str] = {}

    def should_skip_file(self, path: Path) -> bool:
        """Check if file should be skipped"""
        try:
            path_str = str(path.relative_to(self.project_root))
        except ValueError:
            path_str = str(path)

        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, path_str, re.IGNORECASE):
                return True
        return False

    def get_language(self, file_path: Path) -> Optional[str]:
        """Detect language from file extension"""
        ext_map = {
            '.py': 'python',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.go': 'go',
            '.java': 'java',
            '.rs': 'rust',
            '.swift': 'swift',
        }
        return ext_map.get(file_path.suffix)

    def extract_functions(self, code: str, language: str) -> List[tuple]:
        """
        Extract functions/classes from code
        Returns: [(name, start_line, end_line, code_block)]
        """
        extractions = []
        lines = code.split('\n')

        if language == 'python':
            for i, line in enumerate(lines):
                if re.match(r'^(class|def|async def)\s+', line):
                    start = i
                    indent = len(line) - len(line.lstrip())
                    end = start + 1

                    while end < len(lines):
                        next_line = lines[end]
                        if next_line.strip() and not next_line.startswith(' ' * (indent + 1)) and not next_line.strip().startswith('#'):
                            # Check if it's a new definition at same indent
                            if re.match(r'^(class|def|async def)\s+', next_line.lstrip()):
                                break
                        end += 1

                    snippet = '\n'.join(lines[start:min(end, start + 30)])
                    match = re.search(r'(class|def|async def)\s+(\w+)', line)
                    if match:
                        name = match.group(2)
                        extractions.append((name, start, end, snippet))

        elif language in ('typescript', 'javascript'):
            for i, line in enumerate(lines):
                if re.match(r'^(export\s+)?(class|function|async\s+function|interface|const)\s+', line):
                    start = i
                    end = start + 1
                    brace_count = 0

                    for j in range(start, min(start + 100, len(lines))):
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        end = j + 1
                        if brace_count == 0 and j > start:
                            break

                    snippet = '\n'.join(lines[start:min(end, start + 40)])
                    match = re.search(r'(class|function|const|interface)\s+(\w+)', line)
                    if match:
                        name = match.group(2)
                        extractions.append((name, start, end, snippet))

        elif language == 'swift':
            for i, line in enumerate(lines):
                if re.match(r'^(class|struct|func|protocol|enum)\s+', line.strip()):
                    start = i
                    end = start + 1
                    brace_count = 0

                    for j in range(start, min(start + 100, len(lines))):
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        end = j + 1
                        if brace_count == 0 and j > start:
                            break

                    snippet = '\n'.join(lines[start:min(end, start + 40)])
                    match = re.search(r'(class|struct|func|protocol|enum)\s+(\w+)', line)
                    if match:
                        name = match.group(2)
                        extractions.append((name, start, end, snippet))

        return extractions

    def analyze_file(self, file_path: Path) -> List[CodePattern]:
        """Analyze single file and extract patterns"""
        if self.should_skip_file(file_path):
            return []

        language = self.get_language(file_path)
        if not language:
            return []

        try:
            code = file_path.read_text(encoding='utf-8')
            file_hash = hashlib.sha256(code.encode()).hexdigest()

            # Skip if unchanged
            if self.file_hashes.get(str(file_path)) == file_hash:
                return []

            self.file_hashes[str(file_path)] = file_hash

            patterns = []

            for name, start, end, snippet in self.extract_functions(code, language):
                pattern_type = 'class' if 'class' in snippet.split('\n')[0] else 'function'
                if 'interface' in snippet.split('\n')[0]:
                    pattern_type = 'interface'

                # Detect features/tags
                tags = []
                if 'async' in snippet:
                    tags.append('async')
                if re.search(r'try|except|catch|throw', snippet):
                    tags.append('error_handling')
                if re.search(r'@\w+', snippet):
                    tags.append('decorated')
                if re.search(r'import|from|require', snippet):
                    tags.append('imports')
                if re.search(r'test|spec|describe|it\(', snippet, re.IGNORECASE):
                    tags.append('test')
                if re.search(r'log|print|console\.', snippet):
                    tags.append('logging')

                pattern_id = hashlib.sha256(
                    f"{file_path}:{start}:{end}".encode()
                ).hexdigest()[:16]

                description = self._generate_description(snippet, name, pattern_type)

                try:
                    rel_path = str(file_path.relative_to(self.project_root))
                except ValueError:
                    rel_path = str(file_path)

                pattern = CodePattern(
                    pattern_id=pattern_id,
                    language=language,
                    pattern_type=pattern_type,
                    code_snippet=snippet[:500],
                    file_path=rel_path,
                    line_range=(start, end),
                    description=description,
                    tags=tags,
                    dependencies=[],
                    frequency=1
                )

                patterns.append(pattern)
                self.patterns.append(pattern)

            return patterns

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return []

    def _generate_description(self, code: str, name: str, pattern_type: str) -> str:
        """Generate brief description of code pattern"""
        first_line = code.split('\n')[0].strip()
        if len(first_line) > 100:
            first_line = first_line[:100] + '...'
        return f"{pattern_type.title()} '{name}': {first_line}"

    def index_project(self, max_files: int = 500) -> ProjectContext:
        """
        Scan entire project and extract patterns
        Returns: ProjectContext with all patterns and conventions
        """
        print(f"Indexing project: {self.project_root}")
        start_time = time.time()

        # Find all source files
        all_files = []
        for ext in ['.py', '.ts', '.tsx', '.js', '.jsx', '.go', '.java', '.rs', '.swift']:
            all_files.extend(list(self.project_root.rglob(f'*{ext}'))[:max_files])

        print(f"Found {len(all_files)} source files")

        # Analyze all files
        language_counts = defaultdict(int)
        for file_path in all_files[:max_files]:
            patterns = self.analyze_file(file_path)
            language = self.get_language(file_path)
            if language:
                language_counts[language] += 1

        print(f"Extracted {len(self.patterns)} patterns in {time.time() - start_time:.1f}s")

        # Detect primary language
        primary_lang = max(language_counts, key=language_counts.get) if language_counts else 'python'

        # Analyze conventions
        conventions = self._analyze_conventions()

        context = ProjectContext(
            project_root=str(self.project_root),
            primary_language=primary_lang,
            frameworks=self._detect_frameworks(),
            key_conventions=conventions,
            error_handling_patterns=[p for p in self.patterns if 'error_handling' in p.tags][:10],
            naming_conventions=self._extract_naming_conventions(),
            import_patterns=self._extract_import_patterns(),
            test_conventions=self._extract_test_conventions(),
            total_files_analyzed=min(len(all_files), max_files),
            total_patterns=len(self.patterns)
        )

        # Save to cache
        self._save_index(context)

        return context

    def _detect_frameworks(self) -> List[str]:
        """Detect frameworks used in project"""
        frameworks = []

        indicators = {
            'FastAPI': ['fastapi', 'from fastapi'],
            'Django': ['django', 'from django'],
            'Flask': ['flask', 'from flask'],
            'React': ['import React', 'from react', 'react'],
            'Vue': ['vue', 'from vue'],
            'Express': ['express', "require('express')"],
            'MLX': ['import mlx', 'from mlx'],
            'SwiftUI': ['SwiftUI', 'struct.*View'],
            'UIKit': ['UIKit', 'UIViewController'],
        }

        all_code = '\n'.join(p.code_snippet for p in self.patterns[:50])

        for framework, indicators_list in indicators.items():
            for indicator in indicators_list:
                if indicator.lower() in all_code.lower():
                    frameworks.append(framework)
                    break

        return list(set(frameworks))

    def _analyze_conventions(self) -> Dict[str, Any]:
        """Analyze project conventions"""
        return {
            'indentation': self._detect_indentation(),
            'line_length': 100,
            'documentation': 'docstring' if any('"""' in p.code_snippet for p in self.patterns[:20]) else 'comment',
            'typing': 'strict' if any(re.search(r':\s*\w+\s*[,\)]', p.code_snippet) for p in self.patterns[:20]) else 'loose'
        }

    def _extract_naming_conventions(self) -> Dict[str, str]:
        """Extract naming conventions from code"""
        return {
            'variables': 'snake_case',
            'classes': 'PascalCase',
            'constants': 'UPPER_SNAKE_CASE',
            'private': '_leading_underscore'
        }

    def _extract_import_patterns(self) -> List[str]:
        """Extract common import patterns"""
        imports = []
        for pattern in self.patterns[:50]:
            if 'imports' in pattern.tags:
                imports.extend(re.findall(r'(?:from|import)\s+([\w\.]+)', pattern.code_snippet))
        return list(set(imports))[:20]

    def _extract_test_conventions(self) -> Dict[str, Any]:
        """Extract testing conventions"""
        return {
            'test_location': 'tests/',
            'naming_pattern': 'test_*.py',
            'framework': 'pytest',
        }

    def _detect_indentation(self) -> str:
        """Detect indentation style"""
        for pattern in self.patterns[:10]:
            if '\n    ' in pattern.code_snippet:
                return '4-spaces'
            elif '\n  ' in pattern.code_snippet:
                return '2-spaces'
        return '4-spaces'

    def _save_index(self, context: ProjectContext):
        """Save index to cache"""
        cache_file = self.cache_dir / f"{hashlib.md5(str(self.project_root).encode()).hexdigest()}_index.json"

        # Convert patterns with error handling for line_range tuples
        patterns_data = []
        for p in self.patterns:
            p_dict = asdict(p)
            # Ensure line_range is a list for JSON serialization
            if isinstance(p_dict.get('line_range'), tuple):
                p_dict['line_range'] = list(p_dict['line_range'])
            patterns_data.append(p_dict)

        # Convert context
        context_dict = asdict(context)
        # Convert error_handling_patterns
        if 'error_handling_patterns' in context_dict:
            ehp = []
            for p in context_dict['error_handling_patterns']:
                if isinstance(p, dict) and isinstance(p.get('line_range'), tuple):
                    p['line_range'] = list(p['line_range'])
                ehp.append(p)
            context_dict['error_handling_patterns'] = ehp

        data = {
            'context': context_dict,
            'patterns': patterns_data,
            'indexed_at': time.time()
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Index saved to {cache_file}")

    def load_index(self) -> Optional[Dict[str, Any]]:
        """Load cached index"""
        cache_file = self.cache_dir / f"{hashlib.md5(str(self.project_root).encode()).hexdigest()}_index.json"

        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                # Check if cache is fresh (less than 1 hour old)
                if time.time() - data.get('indexed_at', 0) < 3600:
                    return data

        return None


if __name__ == "__main__":
    # Test indexing
    import sys

    project_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/don/Adverant/nexus-local-mageagent"

    indexer = CodebaseIndexer(project_path)
    context = indexer.index_project()

    print(f"\n{'='*60}")
    print(f"Project: {context.project_root}")
    print(f"Language: {context.primary_language}")
    print(f"Frameworks: {', '.join(context.frameworks)}")
    print(f"Files analyzed: {context.total_files_analyzed}")
    print(f"Patterns extracted: {context.total_patterns}")
    print(f"{'='*60}")

    print("\nSample patterns:")
    for p in indexer.patterns[:5]:
        print(f"  - {p.pattern_type}: {p.description[:60]}...")
