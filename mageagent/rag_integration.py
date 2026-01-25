#!/usr/bin/env python3
"""
RAG Integration for MageAgent
Injects project-specific patterns into prompts
"""

import os
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from codebase_indexer import CodebaseIndexer, ProjectContext, CodePattern
from pattern_matcher import PatternMatcher, PatternContextBuilder, SimilarPattern


@dataclass
class ChatMessage:
    """Compatible with server.py ChatMessage"""
    role: str
    content: str


class RAGPromptInjector:
    """Inject RAG context into model prompts"""

    def __init__(self, pattern_context: str, project_conventions: Dict[str, Any]):
        self.pattern_context = pattern_context
        self.project_conventions = project_conventions

    def enrich_messages(
        self,
        messages: List[ChatMessage],
        task_type: str = "coding"
    ) -> List[ChatMessage]:
        """
        Inject RAG context into message chain

        Adds project patterns to system message for context-aware generation.
        """
        enriched = list(messages)

        if not enriched:
            return enriched

        # Find or create system message
        system_idx = None
        for i, msg in enumerate(enriched):
            if msg.role == "system":
                system_idx = i
                break

        # Build enhanced system context
        system_context = self._build_system_context(task_type)

        if system_idx is not None:
            # Append to existing system message
            original = enriched[system_idx].content
            enriched[system_idx] = ChatMessage(
                role="system",
                content=f"{original}\n\n{system_context}"
            )
        else:
            # Insert new system message
            enriched.insert(0, ChatMessage(
                role="system",
                content=system_context
            ))

        return enriched

    def _build_system_context(self, task_type: str) -> str:
        """Build comprehensive system context"""
        parts = [
            "# PROJECT-SPECIFIC CONTEXT",
            "You are generating code for a specific project. Follow these patterns exactly.",
            ""
        ]

        # Add code patterns
        if self.pattern_context:
            parts.append(self.pattern_context)
            parts.append("")

        # Add guidelines
        parts.extend([
            "## GENERATION GUIDELINES",
            "1. Match the project's coding style exactly",
            "2. Follow the error handling patterns shown above",
            "3. Use the same naming conventions",
            "4. Include appropriate documentation",
            ""
        ])

        return '\n'.join(parts)


class MageAgentRAGIntegration:
    """Main RAG integration class for MageAgent server"""

    def __init__(self, codebase_root: str, cache_ttl: int = 3600):
        """
        Initialize RAG integration

        Args:
            codebase_root: Path to project root
            cache_ttl: Cache time-to-live in seconds
        """
        self.codebase_root = Path(codebase_root).resolve()
        self.cache_ttl = cache_ttl

        self.indexer: Optional[CodebaseIndexer] = None
        self.project_context: Optional[ProjectContext] = None
        self.matcher: Optional[PatternMatcher] = None
        self.builder: Optional[PatternContextBuilder] = None
        self.last_index_time: float = 0

        # Initialize on first use
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of indexer and matcher"""
        if self._initialized and (time.time() - self.last_index_time) < self.cache_ttl:
            return

        print(f"[RAG] Initializing for project: {self.codebase_root}")
        start = time.time()

        self.indexer = CodebaseIndexer(str(self.codebase_root))

        # Try to load from cache first
        cached = self.indexer.load_index()

        if cached:
            print("[RAG] Using cached index")
            # Reconstruct from cache
            self.project_context = ProjectContext(**cached['context'])

            # Reconstruct patterns
            patterns = []
            for p_dict in cached['patterns']:
                # Convert line_range back to tuple if needed
                if 'line_range' in p_dict and isinstance(p_dict['line_range'], list):
                    p_dict['line_range'] = tuple(p_dict['line_range'])
                patterns.append(CodePattern(**p_dict))
            self.indexer.patterns = patterns
        else:
            # Fresh index
            self.project_context = self.indexer.index_project()

        self.matcher = PatternMatcher(self.project_context, self.indexer.patterns)
        self.builder = PatternContextBuilder(self.project_context)

        self.last_index_time = time.time()
        self._initialized = True

        print(f"[RAG] Initialized with {len(self.indexer.patterns)} patterns in {time.time() - start:.1f}s")

    async def enrich_generation_request(
        self,
        messages: List[ChatMessage],
        user_prompt: str,
        model_type: str,
        max_patterns: int = 3
    ) -> List[ChatMessage]:
        """
        Enrich a generation request with RAG context

        Args:
            messages: Original message list
            user_prompt: The user's current prompt
            model_type: Which model pattern (hybrid, validated, etc.)
            max_patterns: Max patterns to include

        Returns:
            Enhanced message list with project context
        """
        self._ensure_initialized()

        # Detect task type
        task_type = self._detect_task_type(user_prompt)

        # Skip RAG for non-coding tasks
        if task_type == "general":
            return messages

        # Find similar patterns
        similar_patterns = await self.matcher.find_similar_patterns(
            query=user_prompt,
            task_type=task_type,
            top_k=max_patterns,
            similarity_threshold=0.25
        )

        if not similar_patterns:
            return messages

        # Build context injection
        pattern_context = self.builder.build_context_injection(
            similar_patterns,
            user_prompt,
            max_chars=3000
        )

        # Create injector
        injector = RAGPromptInjector(
            pattern_context=pattern_context,
            project_conventions=asdict(self.project_context) if self.project_context else {}
        )

        # Enrich messages
        enriched = injector.enrich_messages(messages, task_type)

        print(f"[RAG] Enriched with {len(similar_patterns)} patterns for task type: {task_type}")

        return enriched

    def _detect_task_type(self, prompt: str) -> str:
        """Detect what type of code is being requested"""
        prompt_lower = prompt.lower()

        patterns = {
            'function': ['function', 'def ', 'async', 'method', 'fn '],
            'class': ['class', 'interface', 'struct', 'type ', 'model'],
            'error_handling': ['error', 'exception', 'handle', 'try', 'catch'],
            'test': ['test', 'unit test', 'pytest', 'jest', 'spec'],
            'api': ['api', 'endpoint', 'route', 'request', 'response'],
        }

        for task_type, keywords in patterns.items():
            if any(kw in prompt_lower for kw in keywords):
                return task_type

        # Default to function for coding-related requests
        coding_keywords = ['write', 'create', 'implement', 'add', 'build', 'code', 'fix']
        if any(kw in prompt_lower for kw in coding_keywords):
            return 'function'

        return 'general'

    def get_project_summary(self) -> Dict[str, Any]:
        """Get summary of indexed project for health checks"""
        self._ensure_initialized()

        return {
            'project_root': str(self.codebase_root),
            'primary_language': self.project_context.primary_language if self.project_context else 'unknown',
            'frameworks': self.project_context.frameworks if self.project_context else [],
            'total_patterns': len(self.indexer.patterns) if self.indexer else 0,
            'last_indexed': self.last_index_time,
            'cache_ttl': self.cache_ttl,
        }


# Singleton for server integration
_rag_instance: Optional[MageAgentRAGIntegration] = None


def get_rag_integration(project_root: Optional[str] = None) -> Optional[MageAgentRAGIntegration]:
    """Get or create RAG integration singleton"""
    global _rag_instance

    # Get project root from env or parameter
    root = project_root or os.environ.get('MAGEAGENT_PROJECT_ROOT')

    if not root:
        return None

    if _rag_instance is None or str(_rag_instance.codebase_root) != root:
        _rag_instance = MageAgentRAGIntegration(root)

    return _rag_instance


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test RAG integration
        rag = MageAgentRAGIntegration("/Users/don/Adverant/nexus-local-mageagent")

        messages = [
            ChatMessage(role="user", content="Write an async function to load models with error handling")
        ]

        enriched = await rag.enrich_generation_request(
            messages,
            "async function to load models with error handling",
            "mageagent:hybrid"
        )

        print(f"\nOriginal messages: {len(messages)}")
        print(f"Enriched messages: {len(enriched)}")

        for msg in enriched:
            print(f"\n[{msg.role}]")
            print(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)

        # Get summary
        summary = rag.get_project_summary()
        print(f"\n\nProject Summary: {summary}")

    asyncio.run(test())
