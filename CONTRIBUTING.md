# Contributing to Sea Turtle Re-Identification

Thank you for your interest in contributing to this project! This guide will help you get started.

## 🎯 Ways to Contribute

### 1. Research Contributions
- Implement new architectures (Vision Transformers, etc.)
- Improve training methodologies
- Add support for new wildlife species
- Enhance interpretability methods

### 2. Code Contributions
- Bug fixes
- Performance optimizations
- Documentation improvements
- Test coverage expansion

### 3. Dataset Contributions
- Add support for additional wildlife datasets
- Improve data augmentation strategies
- Create benchmark splits for other species

### 4. Documentation Contributions
- Tutorial creation
- API documentation
- Use case examples
- Translating documentation

## 🚀 Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/sea-turtle-reid.git
cd sea-turtle-reid

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_AUTHOR/sea-turtle-reid.git
```

### Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

The `[dev]` extra includes:
- pytest (testing)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)
- pre-commit (git hooks)

## 📝 Development Workflow

### 1. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/issue-description
```

### 2. Make Changes

Follow these guidelines:

#### Code Style

We use **Black** for code formatting:
```bash
black src/ tests/
```

Configuration:
- Line length: 88 characters
- Automatic formatting on save (recommended)

#### Linting

We use **flake8** for linting:
```bash
flake8 src/ tests/
```

#### Type Hints

We use **mypy** for type checking:
```bash
mypy src/
```

Add type hints to new functions:
```python
def compute_metric(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> float:
    """Compute evaluation metric."""
    ...
```

#### Documentation

All functions should have docstrings:
```python
def temporal_split(
    individuals: List[str],
    metadata: Dict[str, Any],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Dict[str, List[str]]:
    """
    Create temporal-aware dataset splits.
    
    Args:
        individuals: List of individual IDs
        metadata: Dictionary mapping image_id to metadata
        ratios: Train, query, gallery split ratios
        
    Returns:
        Dictionary with split assignments
        
    Raises:
        ValueError: If ratios don't sum to 1.0
        
    Example:
        >>> splits = temporal_split(
        ...     individuals=['ind_1', 'ind_2'],
        ...     metadata=metadata,
        ...     ratios=(0.7, 0.15, 0.15)
        ... )
    """
    ...
```

### 3. Write Tests

All new features should include tests:

```python
# tests/test_your_feature.py
import pytest
from src.your_module import your_function

def test_your_function():
    """Test basic functionality."""
    result = your_function(input_data)
    assert result == expected_output

def test_your_function_edge_case():
    """Test edge case."""
    with pytest.raises(ValueError):
        your_function(invalid_input)
```

Run tests:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test function
pytest tests/test_models.py::test_resnet50_creation
```

### 4. Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add Vision Transformer architecture

- Implement ViT-Base and ViT-Large variants
- Add configuration files for both models
- Include tests for model creation and forward pass
- Update documentation with ViT examples"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `style:` Code style changes (formatting)

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Then create a Pull Request on GitHub
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows

### Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures
├── test_models.py        # Model tests
├── test_data.py          # Data loading tests
├── test_evaluation.py    # Metric tests
└── integration/
    └── test_training.py  # Training pipeline tests
```

### Coverage Requirements

- New features: Aim for >80% coverage
- Bug fixes: Include regression test
- Critical paths: Aim for 100% coverage

## 📚 Documentation Guidelines

### README Updates

If your change affects usage:
1. Update relevant sections in README.md
2. Add code examples
3. Update the table of contents if needed

### API Documentation

Document public APIs:
```python
class MyModel(nn.Module):
    """
    Brief description of the model.
    
    This model implements [describe architecture/approach].
    
    Attributes:
        param1: Description of param1
        param2: Description of param2
        
    Example:
        >>> model = MyModel(num_classes=100)
        >>> output = model(input_tensor)
    """
```

### Tutorials

Create Jupyter notebooks for major features:
- Place in `notebooks/`
- Include clear explanations
- Provide complete examples
- Add visualizations

## 🔍 Code Review Process

### What We Look For

✅ **Code Quality**
- Follows project style guidelines
- Well-documented with docstrings
- Includes type hints
- No unnecessary complexity

✅ **Testing**
- Comprehensive test coverage
- Tests pass locally and in CI
- Edge cases considered

✅ **Documentation**
- Clear README updates
- API documentation complete
- Examples provided

✅ **Performance**
- No significant performance regressions
- Efficient algorithms used
- Memory usage considered

### Review Timeline

- Initial review: Within 1 week
- Follow-up reviews: Within 3 days
- Merge: When approved by maintainer

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Test on latest version
3. Verify it's not a local environment issue

### Bug Report Template

```markdown
**Describe the bug**
Clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Load model with '...'
2. Run inference on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.8.10]
- PyTorch version: [e.g., 1.9.0]
- CUDA version: [e.g., 11.1]

**Additional context**
Any other relevant information.
```

## 💡 Feature Requests

### Template

```markdown
**Feature description**
Clear description of the feature.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed solution**
How you envision this feature working.

**Alternatives considered**
Other approaches you've thought about.

**Additional context**
Any other relevant information.
```

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Credited in relevant documentation

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open an Issue
- **Chat**: Join our community channel (link TBD)
- **Email**: your.email@example.com

## 📜 Code of Conduct

### Our Standards

✅ **Be respectful**: Treat everyone with respect
✅ **Be constructive**: Provide helpful feedback
✅ **Be collaborative**: Work together effectively
✅ **Be inclusive**: Welcome diverse perspectives

❌ **Don't**: Harass, discriminate, or be disrespectful

### Enforcement

Violations may result in:
1. Warning
2. Temporary ban
3. Permanent ban

Report issues to: your.email@example.com

## 📋 Checklist Before Submitting PR

- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest tests/`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Docstrings written
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Commits are clear and descriptive

## 🎓 Learning Resources

### Project-Specific
- [Methodology Documentation](docs/METHODOLOGY.md)
- [API Reference](docs/API.md)
- [Full Dissertation](docs/dissertation.pdf)

### General Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Git Best Practices](https://git-scm.com/book/en/v2)
- [Writing Good Commits](https://chris.beams.io/posts/git-commit/)

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

*For questions about this guide, open an issue or contact the maintainers.*
