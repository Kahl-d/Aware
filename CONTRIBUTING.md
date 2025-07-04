# Contributing to Aware

Thank you for your interest in contributing to the Aware project! This document provides guidelines for contributing to our Cultural Capital Theme Classification framework.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Report issues you encounter
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with code changes
- **Documentation**: Improve or add documentation
- **Research**: Contribute to the research aspects
- **Testing**: Help test the framework on different datasets

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/aware.git
   cd aware
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv aware_env
   source aware_env/bin/activate  # On Windows: aware_env\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Development Setup

### Code Style

We follow PEP 8 style guidelines. Please use:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

```bash
# Format code
black .

# Sort imports
isort .

# Check linting
flake8 .
```

### Testing

Run tests before submitting:

```bash
pytest tests/
```

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality:

```bash
pip install pre-commit
pre-commit install
```

## 📝 Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/essay-aware-improvements`
- `bugfix/memory-leak-fix`
- `docs/update-readme`
- `research/new-evaluation-metrics`

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(model): add attention pooling mechanism`
- `fix(data): resolve character span mapping issue`
- `docs(readme): update installation instructions`
- `test(evaluation): add cross-validation tests`

### Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** with clear, focused commits
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run all tests** and ensure they pass
6. **Submit a pull request** with a clear description

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Research contribution
- [ ] Other (please describe)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues
Closes #(issue number)
```

## 🔬 Research Contributions

### Adding New Models

If contributing new model architectures:

1. **Create a new directory** in the project structure
2. **Follow existing patterns** from `multi-base/` or `multi-ea/`
3. **Include comprehensive documentation**
4. **Add evaluation scripts**
5. **Update the main README.md**

### Dataset Contributions

For new datasets:

1. **Ensure proper anonymization**
2. **Follow the existing data format**
3. **Include data collection methodology**
4. **Add data validation scripts**

### Evaluation Metrics

When adding new evaluation metrics:

1. **Implement in a modular way**
2. **Add unit tests**
3. **Document the metric's purpose**
4. **Include baseline comparisons**

## 🐛 Reporting Issues

### Bug Reports

Use the issue template and include:

- **Clear description** of the problem
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Error messages** and stack traces
- **Minimal example** if possible

### Feature Requests

Include:

- **Problem description**
- **Proposed solution**
- **Use cases**
- **Implementation suggestions** (if any)

## 📚 Documentation

### Code Documentation

- Use **docstrings** for all functions and classes
- Follow **Google style** docstring format
- Include **type hints** where appropriate
- Add **examples** for complex functions

### User Documentation

- Keep **README.md** up to date
- Add **usage examples**
- Include **troubleshooting guides**
- Update **installation instructions**

## 🎯 Areas for Contribution

### High Priority

- **Performance optimization** for large datasets
- **Additional evaluation metrics**
- **Cross-lingual support**
- **Real-time inference capabilities**

### Medium Priority

- **Web interface** for model testing
- **Additional pre-trained models**
- **Data augmentation techniques**
- **Interpretability tools**

### Research Opportunities

- **New Cultural Capital frameworks**
- **Multi-modal analysis** (text + metadata)
- **Longitudinal studies**
- **Cross-institutional comparisons**

## 🏆 Recognition

Contributors will be recognized in:

- **README.md** acknowledgments
- **Research papers** (for significant contributions)
- **Release notes**
- **Contributor hall of fame**

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: kkhan@sfsu.edu for research collaboration

## 📄 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be **respectful** and **inclusive**
- Focus on **constructive feedback**
- Respect **diverse perspectives**
- Follow **professional standards**

## 📋 Contributor License Agreement

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to making education more equitable through AI! 🎓✨ 