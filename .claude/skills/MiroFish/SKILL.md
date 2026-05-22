```markdown
# MiroFish Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and conventions used in the MiroFish Python codebase. You will learn how to structure files, write imports and exports, follow commit message conventions, and organize tests. These patterns help maintain consistency and readability across the project.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - Example: `miro_fish_module.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import process_fish
    ```

### Export Style
- Use **named exports** by explicitly listing public objects in `__all__`.
  - Example:
    ```python
    __all__ = ['Fish', 'process_fish']
    ```

### Commit Messages
- Use **conventional commits** with the `feat` prefix for new features.
- Keep commit messages concise (average: 48 characters).
  - Example:
    ```
    feat: add fish movement simulation
    ```

## Workflows

### Feature Development
**Trigger:** When adding a new feature  
**Command:** `/feature-dev`

1. Create a new Python module using snake_case naming.
2. Implement the feature using relative imports as needed.
3. Export public objects using `__all__`.
4. Write or update corresponding test files (`*.test.*`).
5. Commit changes with a message like:  
   `feat: <short description of feature>`

### Testing
**Trigger:** When verifying code functionality  
**Command:** `/run-tests`

1. Locate or create test files matching `*.test.*`.
2. Run tests using your preferred Python test runner (framework is unspecified).
3. Ensure all tests pass before merging or deploying.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `fish_behavior.test.py`)
- The specific testing framework is not defined; use standard Python testing practices.
- Place tests alongside or near the modules they cover.

  Example test file:
  ```python
  # fish_behavior.test.py
  from .fish_behavior import Fish

  def test_fish_swim():
      fish = Fish()
      assert fish.swim() == "swimming"
  ```

## Commands
| Command        | Purpose                                   |
|----------------|-------------------------------------------|
| /feature-dev   | Start a new feature development workflow   |
| /run-tests     | Run all test files in the codebase         |
```
