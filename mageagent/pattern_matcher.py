#!/usr/bin/env python3
"""
Pattern Matcher - Find similar code patterns using semantic similarity
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from codebase_indexer import CodePattern, ProjectContext


@dataclass
class SimilarPattern:
    """A matched pattern with similarity info"""
    pattern: CodePattern
    similarity_score: float  # 0-1
    reason: str
    suggestion: str


class PatternMatcher:
    """Match code patterns by semantic similarity"""

    # Keywords that boost matching
    BOOST_KEYWORDS = {
        'async': ['await', 'asyncio', 'concurrent', 'parallel'],
        'error_handling': ['try', 'except', 'catch', 'throw', 'error', 'exception'],
        'api': ['request', 'response', 'endpoint', 'route', 'http'],
        'database': ['query', 'sql', 'db', 'model', 'schema'],
        'test': ['test', 'assert', 'mock', 'fixture', 'pytest'],
        'validation': ['validate', 'check', 'verify', 'schema'],
    }

    def __init__(self, project_context: ProjectContext, patterns: List[CodePattern]):
        self.project_context = project_context
        self.patterns = patterns
        self._build_keyword_index()

    def _build_keyword_index(self):
        """Build inverted index of keywords to patterns"""
        self.keyword_to_patterns: Dict[str, List[int]] = {}

        for i, pattern in enumerate(self.patterns):
            # Extract keywords from pattern
            text = (
                pattern.code_snippet.lower() + ' ' +
                pattern.description.lower() + ' ' +
                ' '.join(pattern.tags)
            )

            words = set(re.findall(r'\b\w+\b', text))

            for word in words:
                if word not in self.keyword_to_patterns:
                    self.keyword_to_patterns[word] = []
                self.keyword_to_patterns[word].append(i)

    async def find_similar_patterns(
        self,
        query: str,
        task_type: str = "function",
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[SimilarPattern]:
        """
        Find patterns similar to the query

        Args:
            query: Description of what to generate
            task_type: Type of code pattern (function, class, error_handling)
            top_k: Return top K matches
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar patterns ranked by relevance
        """
        # Filter by pattern type first
        relevant_patterns = [
            p for p in self.patterns
            if task_type in p.pattern_type or task_type in p.tags
        ]

        if not relevant_patterns:
            relevant_patterns = self.patterns

        # Score each pattern
        scored = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))

        for pattern in relevant_patterns:
            score = self._compute_similarity(query_words, query_lower, pattern)

            if score >= similarity_threshold:
                similar = SimilarPattern(
                    pattern=pattern,
                    similarity_score=score,
                    reason=self._explain_match(query_words, pattern),
                    suggestion=self._generate_suggestion(pattern)
                )
                scored.append(similar)

        # Sort by score
        scored.sort(key=lambda x: x.similarity_score, reverse=True)

        return scored[:top_k]

    def _compute_similarity(
        self,
        query_words: set,
        query_lower: str,
        pattern: CodePattern
    ) -> float:
        """Compute similarity score between query and pattern"""

        score = 0.0

        pattern_text = (
            pattern.code_snippet.lower() + ' ' +
            pattern.description.lower() + ' ' +
            ' '.join(pattern.tags)
        )
        pattern_words = set(re.findall(r'\b\w+\b', pattern_text))

        # Word overlap
        overlap = query_words & pattern_words
        if overlap:
            score += len(overlap) * 0.05

        # Tag matching (high weight)
        for tag in pattern.tags:
            if tag in query_lower:
                score += 0.2

        # Pattern type matching
        if pattern.pattern_type in query_lower:
            score += 0.15

        # Boost for specific concepts
        for concept, keywords in self.BOOST_KEYWORDS.items():
            concept_in_query = concept in query_lower or any(k in query_lower for k in keywords)
            concept_in_pattern = concept in pattern.tags or any(k in pattern_text for k in keywords)

            if concept_in_query and concept_in_pattern:
                score += 0.15

        # Length normalization (prefer patterns that aren't too short or long)
        snippet_len = len(pattern.code_snippet)
        if 100 < snippet_len < 400:
            score += 0.05

        return min(score, 1.0)

    def _explain_match(self, query_words: set, pattern: CodePattern) -> str:
        """Explain why this pattern matches"""
        reasons = []

        if pattern.pattern_type in str(query_words):
            reasons.append(f"Type: {pattern.pattern_type}")

        for tag in pattern.tags:
            if tag in str(query_words):
                reasons.append(f"Feature: {tag}")

        if pattern.frequency > 3:
            reasons.append("Commonly used pattern")

        # Language match
        if pattern.language == self.project_context.primary_language:
            reasons.append(f"Same language: {pattern.language}")

        return "; ".join(reasons) if reasons else "Similar code structure"

    def _generate_suggestion(self, pattern: CodePattern) -> str:
        """Generate suggestion for using this pattern"""
        return f"Reference: {pattern.file_path}:{pattern.line_range[0]}"


class PatternContextBuilder:
    """Build rich context from matched patterns"""

    def __init__(self, project_context: ProjectContext):
        self.project_context = project_context

    def build_context_injection(
        self,
        similar_patterns: List[SimilarPattern],
        query: str,
        max_chars: int = 4000
    ) -> str:
        """
        Build prompt injection text with matched patterns

        Returns:
            Formatted context string to inject into prompt
        """
        if not similar_patterns:
            return ""

        lines = [
            "## PROJECT CODE PATTERNS",
            "Follow these existing patterns from the project:",
            ""
        ]

        total_chars = len('\n'.join(lines))

        for i, match in enumerate(similar_patterns[:5], 1):
            pattern = match.pattern

            pattern_block = [
                f"### Pattern {i}: {pattern.pattern_type.upper()} ({match.similarity_score*100:.0f}% match)",
                f"**File**: `{pattern.file_path}` line {pattern.line_range[0]}",
                f"**Tags**: {', '.join(pattern.tags) if pattern.tags else 'none'}",
                "",
                "```" + pattern.language,
                pattern.code_snippet,
                "```",
                ""
            ]

            block_text = '\n'.join(pattern_block)

            if total_chars + len(block_text) > max_chars:
                break

            lines.extend(pattern_block)
            total_chars += len(block_text)

        # Add conventions section
        if self.project_context.key_conventions:
            conv_lines = [
                "## PROJECT CONVENTIONS",
                f"- **Language**: {self.project_context.primary_language}",
                f"- **Naming**: {self.project_context.naming_conventions.get('variables', 'snake_case')}",
                f"- **Frameworks**: {', '.join(self.project_context.frameworks) if self.project_context.frameworks else 'None detected'}",
                ""
            ]
            lines.extend(conv_lines)

        return '\n'.join(lines)


if __name__ == "__main__":
    import asyncio
    from codebase_indexer import CodebaseIndexer

    async def test():
        # Index project
        indexer = CodebaseIndexer("/Users/don/Adverant/nexus-local-mageagent")
        context = indexer.index_project()

        # Create matcher
        matcher = PatternMatcher(context, indexer.patterns)

        # Test queries
        queries = [
            ("async function to fetch data with error handling", "function"),
            ("class for managing state", "class"),
            ("error handling pattern", "error_handling"),
        ]

        for query, task_type in queries:
            print(f"\nQuery: '{query}'")
            similar = await matcher.find_similar_patterns(query, task_type, top_k=3)

            print(f"Found {len(similar)} matches:")
            for match in similar:
                print(f"  - {match.pattern.pattern_type}: {match.similarity_score*100:.0f}% - {match.pattern.file_path}")

            # Build context
            builder = PatternContextBuilder(context)
            injection = builder.build_context_injection(similar, query)
            print(f"\nContext length: {len(injection)} chars")

    asyncio.run(test())