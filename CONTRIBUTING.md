# Contributing to FlashCls

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/FlashVision/FlashCls.git
cd FlashCls
pip install -e ".[dev,all]"
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run lint: `ruff check flashcls/`
4. Run tests: `flashcls check`
5. Commit and push
6. Open a Pull Request

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting (line length: 120)
- Type hints are encouraged
- Docstrings for all public functions (Google style)
- No hardcoded file paths — use relative or configurable paths

## Adding a New Solution

1. Create `flashcls/solutions/your_solution.py`
2. Follow the existing pattern: accept `predictor` instance
3. Implement core classification logic
4. Add to `flashcls/solutions/__init__.py`

## Commit Messages

Use clear, descriptive messages:
- `Add image tagging solution`
- `Fix label smoothing edge case`
- `Update README with training examples`

## Reporting Issues

- Use GitHub Issues
- Include: Python version, PyTorch version, GPU info, error traceback
- Run `flashcls settings` and paste the output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
