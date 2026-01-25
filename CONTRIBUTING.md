# Contributing to Gmail Takeout RAG

Thank you for considering contributing to this project!

## How to Contribute

### For External Contributors (Fork Method)

1. **Fork the repository** to your GitHub account
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gmail-takeout-rag.git
   cd gmail-takeout-rag
   ```

3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** from your fork to the main repository

### For Collaborators (Branch Method)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shahfazal/gmail-takeout-rag.git
   cd gmail-takeout-rag
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

4. **Push your branch**:
   ```bash
   git push -u origin feature/your-feature-name
   ```

5. **Open a Pull Request** on GitHub

## Branch Protection Rules

The `main` branch is protected with the following rules:

- **Pull requests required** - No direct pushes to main
- **No force pushes** - History cannot be rewritten
- **No branch deletion** - Main branch cannot be deleted

This means **all changes must go through pull requests**, even for repository maintainers.

## Pull Request Guidelines

- Write clear, descriptive commit messages
- Keep PRs focused on a single feature or fix
- Update documentation if you're changing functionality
- Test your changes locally before submitting
- Reference any related issues in your PR description

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise

## Testing

Before submitting a PR:

1. Test your changes locally
2. Ensure the RAG pipeline still works
3. Verify no personal data or API keys are included

## Questions?

Feel free to open an issue for any questions or discussions about contributing.
