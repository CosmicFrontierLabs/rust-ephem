# Contributing to rust-ephem

Thank you for your interest in contributing to `rust-ephem`! This document provides guidelines and instructions for setting up your development environment and contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Development Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or later**: [Download Python](https://www.python.org/downloads/)
- **Rust 1.70 or later**: [Install Rust](https://rustup.rs/)
- **Git**: [Install Git](https://git-scm.com/downloads)

### Setting Up Your Development Environment

1. **Fork and Clone the Repository**

   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/rust-ephem.git
   cd rust-ephem
   ```

2. **Create a Virtual Environment**

   ```bash
   # Create a virtual environment
   python -m venv .venv

   # Activate the virtual environment
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

   Note that we like to use [`uv`](https://docs.astral.sh/uv/) for handling
   installs and virtual environments. Here's how do do the above with `uv`. The
   only real difference from above is use `uv venv` command to create the virtual
   environment. For `pip` commands replace with `uv pip`.

3. **Install Development Dependencies**

   ```bash
   # Install the package in editable mode with all dependencies
   pip install -e ".[dev,test,docs]"
   ```

4. **Install Prek Hooks**

   This project uses [prek](https://github.com/j178/prek). `prek` is a Rust
   replacement for `pre-commit`, software that runs checks on code at time of
   `git commit`. These hooks to ensure code quality and consistency. Pre-commit
   hooks automatically run before each commit to check formatting, linting,
   and other quality checks.

   ```bash
   # Install pre-commit hooks
   prek install
   ```

   The pre-commit configuration includes:

   - **Rust formatting** with `cargo fmt`
   - **Rust linting** with `clippy`
   - **Python linting** with `ruff`
   - **Python formatting** with `ruff format`
   - **YAML validation**
   - **Prevention of commits to main branch**
   - **Trailing whitespace removal**
   - **Large file detection**

5. **Build the Rust Extension**

   ```bash
   # Build the Rust extension in debug mode
   maturin develop

   # Or build in release mode (faster, but slower to compile)
   maturin develop --release
   ```

### Verifying Your Setup

Run the test suite to ensure everything is working correctly:

```bash
pytest
```

## Development Workflow

### Making Changes

1. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make Your Changes**

   - Edit Python code in `rust_ephem/`
   - Edit Rust code in `src/`
   - Add or update tests in `tests/`
   - Update documentation as needed

3. **Rebuild After Rust Changes**

   If you modify Rust code, rebuild the extension:

   ```bash
   maturin develop
   ```

4. **Run Tests**

   ```bash
   # Run all tests
   pytest

   # Run specific test file
   pytest tests/test_specific.py

   # Run with coverage
   pytest --cov=rust_ephem --cov-report=html
   ```

5. **Commit Your Changes**

   Pre-commit hooks will automatically run when you commit:

   ```bash
   git add .
   git commit -m "change(code): brief description of your changes"
   ```

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for
commit messages please.

If the pre-commit hooks fail, fix the issues and commit again. You can also run the hooks manually:

```bash
prek run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_constraints.py

# Run tests matching a pattern
pytest -k "test_sun"

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=rust_ephem --cov-report=term-missing
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_<functionality>_<scenario>`
- Test both Python and Rust components
- Include docstrings explaining what each test validates
- Use fixtures for common setup (see `tests/conftest.py`)
- Any new functionality should be covered by unit tests as much as possible

### Running Rust Tests

In addition to Python integration tests, you can run Rust unit tests directly:

```bash
# Run all Rust tests
cargo test

# Run specific test module
cargo test test_module_name

# Run tests with output
cargo test -- --nocapture

# Run tests in release mode (faster execution)
cargo test --release
```

Rust tests are located in the same files as the code they test (using `#[cfg(test)]` modules) or in separate files in the `tests/` directory at the Rust project root. When contributing Rust code:

- Add unit tests in `#[cfg(test)]` modules within the same file
- Add integration tests in `.rs` files in the `tests/` directory
- Test edge cases and error conditions
- Use descriptive test names: `test_<functionality>_<scenario>`
- Document complex test setups with comments

## Code Quality

### Python Code Style

We use [ruff](https://docs.astral.sh/ruff/) for Python linting and formatting:

```bash
# Lint Python code
ruff check .

# Format Python code
ruff format .

# Fix auto-fixable issues
ruff check --fix .
```

### Rust Code Style

We use standard Rust tooling:

```bash
# Format Rust code
cargo fmt

# Run clippy linter
cargo clippy --all-targets --all-features

# Build and check
cargo check
```

### Type Checking

Use `mypy` for type checking Python code:

```bash
mypy rust_ephem/
```

## Pull Request Process

1. **Update Documentation**

   - Update docstrings for any new or modified functions
   - Update README.md if adding new features
   - Update relevant documentation in `docs/`

2. **Ensure All Tests Pass**

   ```bash
   pytest
   ```

3. **Ensure Pre-commit Hooks Pass**

   ```bash
   prek run --all-files
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**

   - Go to the [rust-ephem repository](https://github.com/CosmicFrontierLabs/rust-ephem)
   - Click "New Pull Request"
   - Select your fork and branch
   - Provide a clear description of your changes
   - Link any related issues

6. **Code Review**

   - Respond to feedback from reviewers
   - Make requested changes
   - Push updates to your branch (the PR will update automatically)

### Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Write clear, descriptive commit messages
- Include tests for new functionality
- Update documentation as needed
- Ensure CI checks pass

## Documentation

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html  # macOS
# or
xdg-open _build/html/index.html  # Linux
```

### Documentation Guidelines

- Use clear, concise language
- Include code examples for new features
- Update API documentation for changed functions
- Add examples to the appropriate `.rst` files in `docs/`

## Performance Considerations

`rust-ephem` is designed for speed, so please don't slow it down. When
contributing performance-critical code:

1. **Profile if necessary**

   ```bash
   # Python profiling
   python -m cProfile -o profile.stats your_script.py

   # Rust profiling
   cargo build --release
   # Use appropriate profiling tools for your platform
   ```

2. **Document performance improvements**

   - Explain algorithmic changes that improve performance
   - Include profiling results if relevant

3. **Benchmark modified code**

   - In python notebooks use `%timeit` to compare code speed.

## Getting Help

- **Bugs / Feature Requests**: Open an [Issue](https://github.com/CosmicFrontierLabs/rust-ephem/issues)
- **Security**: Email <jamie@cosmicfrontier.org>

## Additional Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Documentation](https://www.maturin.rs/)
- [pytest Documentation](https://docs.pytest.org/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)

Thank you for contributing to rust-ephem! 🚀
