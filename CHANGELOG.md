# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- LangGraph backend integration
- CrewAI backend integration
- Google A2A backend integration
- Enhanced documentation with tutorials
- Performance benchmarks

## [0.1.0] - 2025-11-22

### Added
- Initial public release
- Core workflow orchestration framework
- Multi-backend architecture with plugin system
- Native backend implementation (`WorkflowExecutor`)
- Chaining API with pipe operator (`|`)
- `ChainableNode` base class for fluent composition
- `FunctionNode` for wrapping functions as nodes
- `PassthroughNode` for debugging workflows
- Streaming support with `stream()` method
- SSE (Server-Sent Events) integration
- `WorkflowContext` for shared state management
- `NodeResult` for structured execution results
- CEL expression support (optional dependency)
- Comprehensive test suite (84 tests)
- Full type hints and `py.typed` marker
- 6 example workflows demonstrating features
- Documentation with MkDocs/Sphinx support

### Features
- **Type-Safe**: Full Pydantic v2 support with validation
- **Async-First**: Built on asyncio for high performance
- **Modular**: Composable nodes with clear interfaces
- **Observable**: Built-in execution tracking and metadata
- **Flexible**: Conditional branching, loops, parallel execution
- **Extensible**: Backend registry for custom implementations

### Node Types
- `BaseNode` - Foundation for all nodes
- `ChainableNode` - Enables pipe operator chaining
- `FunctionNode` - Wraps sync/async functions
- `PassthroughNode` - Identity transform for debugging

### API Methods
- `run(input_data)` - Execute workflow synchronously
- `stream(input_data)` - Stream results asynchronously
- `pipe()` / `|` operator - Chain nodes fluently

### Backends
- Native backend (default) - Pure Python implementation
- Backend registry system for plugins
- Protocol-based interface for custom backends

### Development
- Black code formatting (line-length: 120)
- Ruff linting with strict rules
- mypy type checking
- pytest with asyncio support
- Coverage reporting
- CI/CD ready configuration

### Documentation
- Comprehensive README with examples
- API reference structure
- Example workflows:
  - Basic chat bot
  - Streaming chat
  - RAG pipeline
  - Advanced features demo
  - HITL (Human-in-the-Loop)
  - Final features showcase

### Comparison
- Detailed comparison with LangChain/LangGraph
- Migration path documentation
- Use case guidelines

## [0.0.1] - 2024-11-XX (Internal)

### Added
- Initial internal prototype
- Basic workflow execution
- Simple node chaining
- Proof of concept for multi-backend architecture

---

## Version Naming Convention

- **Major (X.0.0)**: Breaking API changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible

## Release Process

1. Update CHANGELOG.md with version and date
2. Update version in pyproject.toml
3. Run full test suite
4. Create git tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
5. Push tag: `git push origin vX.Y.Z`
6. Build package: `python -m build`
7. Upload to PyPI: `twine upload dist/*`

---

[Unreleased]: https://github.com/nadoo-ai/nadoo-flow-core/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nadoo-ai/nadoo-flow-core/releases/tag/v0.1.0
[0.0.1]: https://github.com/nadoo-ai/nadoo-flow-core/releases/tag/v0.0.1
