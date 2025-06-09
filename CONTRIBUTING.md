# 🤝 Contributing to Kelpie Carbon v1

Thank you for your interest in contributing to Kelpie Carbon v1! This guide will help you get started with contributing to our kelp forest carbon sequestration assessment application.

## **Table of Contents**
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

---

## **Code of Conduct**

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive environment for all contributors.

### **Our Pledge**
- Be respectful and inclusive
- Focus on what is best for the community
- Show empathy towards other community members
- Accept constructive criticism gracefully

---

## **Getting Started**

### **Prerequisites**
- **Python 3.12+**: Latest Python version
- **Poetry**: For dependency management
- **Git**: For version control
- **Modern Browser**: For testing the web interface
- **Basic Knowledge**: Python, JavaScript, web development

### **Fork and Clone**
```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/kelpie-carbon-v1.git
cd kelpie-carbon-v1

# 3. Add upstream remote
git remote add upstream https://github.com/original-org/kelpie-carbon-v1.git

# 4. Verify remotes
git remote -v
```

### **Development Setup**
```bash
# Install dependencies
poetry install --with dev

# Activate virtual environment
poetry shell

# Install pre-commit hooks
poetry run pre-commit install

# Run tests to ensure everything works
poetry run pytest

# Start development server
poetry run uvicorn src.kelpie_carbon_v1.api.main:app --reload
```

### **Verify Installation**
```bash
# Check code formatting
poetry run black --check src/ tests/

# Check linting
poetry run flake8 src/ tests/

# Check type hints
poetry run mypy src/

# Run test suite
poetry run pytest -v
```

---

## **Development Workflow**

### **Branch Strategy**
We use the **GitHub Flow** branching model:

```bash
# 1. Create feature branch from main
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name

# 2. Make changes and commit
git add .
git commit -m "Add feature: description"

# 3. Push to your fork
git push origin feature/your-feature-name

# 4. Create pull request on GitHub
```

### **Branch Naming Convention**
- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements
- `performance/description` - Performance improvements

### **Commit Message Format**
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### **Types**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

#### **Examples**
```bash
# Good commit messages
feat(api): add kelp biomass estimation endpoint
fix(imagery): resolve RGB composite generation issue
docs(readme): update installation instructions
test(integration): add comprehensive workflow tests

# Bad commit messages
feat: stuff
fix: bug
update readme
```

---

## **Coding Standards**

### **Python Code Style**

#### **Formatting**
We use **Black** for code formatting:
```bash
# Format code
poetry run black src/ tests/

# Check formatting
poetry run black --check src/ tests/
```

#### **Linting**
We use **Flake8** for linting:
```bash
# Run linting
poetry run flake8 src/ tests/

# Configuration in .flake8 file
```

#### **Type Hints**
We use **mypy** for type checking:
```bash
# Run type checking
poetry run mypy src/

# Configuration in myproject.toml [tool.mypy] section
```

#### **Import Organization**
Use **isort** for import organization:
```bash
# Sort imports
poetry run isort src/ tests/

# Check import sorting
poetry run isort --check-only src/ tests/
```

### **Python Best Practices**

#### **Function Documentation**
Use Google-style docstrings:
```python
def calculate_kelp_biomass(
    spectral_data: np.ndarray,
    depth_mask: np.ndarray
) -> Dict[str, float]:
    """Calculate kelp biomass from spectral data.
    
    Args:
        spectral_data: Multi-band spectral imagery array
        depth_mask: Boolean mask for appropriate depth range
        
    Returns:
        Dictionary containing biomass estimates and confidence metrics
        
    Raises:
        ValueError: If spectral_data dimensions are invalid
        
    Example:
        >>> biomass = calculate_kelp_biomass(data, mask)
        >>> print(biomass['total_biomass'])
        156.7
    """
    # Implementation here
    pass
```

#### **Error Handling**
Use specific exception types and meaningful messages:
```python
# Good
if not isinstance(coordinates, dict):
    raise TypeError(
        f"Expected coordinates as dict, got {type(coordinates)}"
    )

if coordinates['lat'] < -90 or coordinates['lat'] > 90:
    raise ValueError(
        f"Latitude {coordinates['lat']} out of valid range [-90, 90]"
    )

# Bad
if bad_condition:
    raise Exception("Something went wrong")
```

#### **Logging**
Use the logging module, not print statements:
```python
import logging

logger = logging.getLogger(__name__)

def process_satellite_data(scene_id: str) -> None:
    """Process satellite data for given scene."""
    logger.info(f"Processing satellite scene: {scene_id}")
    
    try:
        # Processing logic
        logger.debug("Satellite data processed successfully")
    except Exception as e:
        logger.error(f"Failed to process scene {scene_id}: {e}")
        raise
```

### **JavaScript Code Style**

#### **ES6+ Features**
Use modern JavaScript features:
```javascript
// Use const/let instead of var
const apiEndpoint = '/api/imagery';
let analysisId = null;

// Use arrow functions
const loadLayer = async (layerType) => {
    try {
        const response = await fetch(`${apiEndpoint}/${analysisId}/${layerType}`);
        return await response.blob();
    } catch (error) {
        console.error(`Failed to load ${layerType}:`, error);
        throw error;
    }
};

// Use template literals
const imageUrl = `${apiEndpoint}/${analysisId}/rgb`;

// Use destructuring
const { lat, lng } = coordinates;
```

#### **Function Documentation**
Use JSDoc comments:
```javascript
/**
 * Load satellite imagery layers progressively
 * @param {string} analysisId - Unique analysis identifier
 * @param {Array<string>} layers - Array of layer types to load
 * @returns {Promise<Map>} Map of layer types to loaded images
 */
async function loadLayersProgressively(analysisId, layers) {
    // Implementation here
}
```

#### **Error Handling**
Use try-catch with meaningful error messages:
```javascript
// Good
try {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return await response.json();
} catch (error) {
    console.error('Failed to fetch data:', error);
    showUserError('Unable to load satellite data. Please try again.');
    throw error;
}
```

---

## **Testing Guidelines**

### **Test Structure**
Organize tests by component and type:
```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_api.py
│   ├── test_imagery.py
│   └── test_models.py
├── integration/             # Component interaction tests
│   ├── test_workflow.py
│   └── test_api_integration.py
├── performance/             # Performance and load tests
│   └── test_performance.py
└── fixtures/               # Test data and utilities
    ├── sample_data.py
    └── mock_responses.py
```

### **Testing Principles**

#### **Test Naming**
Use descriptive test names:
```python
# Good
def test_kelp_detection_with_clear_water_conditions():
def test_biomass_calculation_handles_missing_data():
def test_api_returns_422_for_invalid_coordinates():

# Bad  
def test_kelp_function():
def test_api():
def test_error():
```

#### **Arrange-Act-Assert Pattern**
Structure tests clearly:
```python
def test_spectral_index_calculation():
    # Arrange
    red_band = np.array([[0.1, 0.2], [0.3, 0.4]])
    nir_band = np.array([[0.5, 0.6], [0.7, 0.8]])
    
    # Act
    ndvi = calculate_ndvi(red_band, nir_band)
    
    # Assert
    expected = (nir_band - red_band) / (nir_band + red_band)
    np.testing.assert_array_almost_equal(ndvi, expected)
```

#### **Test Independence**
Each test should be independent:
```python
class TestSatelliteAPI:
    def setup_method(self):
        """Setup fresh state for each test."""
        self.client = TestClient(app)
        self.test_analysis_id = "test-12345"
    
    def test_rgb_endpoint(self):
        # Test doesn't depend on other tests
        pass
    
    def test_metadata_endpoint(self):
        # Independent test
        pass
```

### **Coverage Requirements**
- **Minimum Coverage**: 80% overall
- **Critical Paths**: 95% coverage for core functionality
- **New Code**: 90% coverage for new features

```bash
# Run with coverage
poetry run pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### **Performance Tests**
Mark slow tests appropriately:
```python
@pytest.mark.slow
def test_large_dataset_processing():
    """Test processing with large satellite datasets."""
    # This test takes more than 10 seconds
    pass

@pytest.mark.performance
def test_api_response_time():
    """Test API response times under load."""
    pass
```

---

## **Documentation**

### **Documentation Types**

#### **Code Documentation**
- **Docstrings**: For all public functions and classes
- **Inline Comments**: For complex logic
- **Type Hints**: For all function parameters and returns

#### **User Documentation**
- **README.md**: Project overview and quick start
- **USER_GUIDE.md**: Comprehensive user instructions
- **API_REFERENCE.md**: Complete API documentation

#### **Developer Documentation**
- **ARCHITECTURE.md**: System design and components
- **TESTING_GUIDE.md**: Testing strategies and guidelines
- **DEPLOYMENT_GUIDE.md**: Deployment instructions

### **Writing Guidelines**

#### **Style**
- Use clear, concise language
- Write for your target audience
- Include examples and code snippets
- Use proper markdown formatting

#### **Structure**
- Use descriptive headings
- Include table of contents for long documents
- Add cross-references between related sections
- Include troubleshooting sections

#### **Code Examples**
Always test code examples:
```python
# ✅ Good: Working example with context
def analyze_kelp_forest(coordinates: Dict[str, float]) -> Dict[str, Any]:
    """Analyze kelp forest at given coordinates.
    
    Example:
        >>> result = analyze_kelp_forest({"lat": 34.4140, "lng": -119.8489})
        >>> print(result['biomass'])
        156.7
    """
    # Implementation here
```

---

## **Pull Request Process**

### **Before Creating PR**

#### **Self-Review Checklist**
- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated (if needed)
- [ ] No debugging code left in
- [ ] Commits are well-structured and descriptive

#### **Code Quality Checks**
```bash
# Run all quality checks
poetry run black --check src/ tests/
poetry run flake8 src/ tests/
poetry run mypy src/
poetry run pytest --cov=src
```

### **PR Template**
When creating a PR, use this template:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that breaks existing functionality)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed
- [ ] Edge cases considered

## Screenshots/Videos
(If applicable, add screenshots or videos demonstrating the changes)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is properly commented
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All checks pass

## Related Issues
Closes #(issue number)
```

### **Review Process**

#### **Automated Checks**
All PRs must pass:
- Code formatting (Black)
- Linting (Flake8)
- Type checking (mypy)
- Test suite (pytest)
- Security scanning (Bandit)

#### **Human Review**
PRs require approval from:
- **1 maintainer** for minor changes
- **2 maintainers** for major changes
- **Domain expert** for specialized changes

#### **Review Guidelines**
**For Reviewers:**
- Focus on functionality, not style (automated tools handle style)
- Check for security issues
- Verify tests are comprehensive
- Ensure documentation is clear
- Be constructive and respectful

**For Contributors:**
- Respond to feedback promptly
- Ask questions if feedback is unclear
- Make requested changes or explain why not
- Keep PR scope focused

---

## **Issue Reporting**

### **Bug Reports**
Use the bug report template:

```markdown
**Bug Description**
A clear and concise description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
- Browser: [e.g. Chrome 96]
- Operating System: [e.g. Windows 10]
- Application Version: [e.g. 1.0.0]

**Additional Context**
Any other context about the problem.
```

### **Feature Requests**
Use the feature request template:

```markdown
**Is your feature request related to a problem?**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

### **Issue Labels**
We use these labels to categorize issues:
- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements or additions to documentation
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `question` - Further information is requested
- `wontfix` - This will not be worked on

---

## **Community**

### **Communication Channels**
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions and reviews

### **Getting Help**
- **Documentation**: Check existing documentation first
- **Search Issues**: Look for similar issues or discussions
- **Create Issue**: If you can't find an answer, create a new issue
- **Be Patient**: Maintainers are volunteers with limited time

### **Recognition**
We recognize contributors through:
- **Contributor List**: All contributors listed in README
- **Release Notes**: Significant contributions mentioned
- **GitHub Stars**: Star the repository to show support

---

## **Development Tips**

### **Productive Development**

#### **IDE Setup**
Recommended VS Code extensions:
- Python (Microsoft)
- Black Formatter
- Flake8
- mypy
- Prettier (for JavaScript)
- Live Server (for frontend testing)

#### **Environment Variables**
Create `.env` file for local development:
```bash
KELPIE_ENV=development
KELPIE_DEBUG=true
KELPIE_LOG_LEVEL=debug
```

#### **Debugging**
Use Python debugger for complex issues:
```python
import pdb; pdb.set_trace()
```

For JavaScript debugging:
```javascript
console.log('Debug info:', variable);
debugger; // Browser will pause here
```

### **Common Development Tasks**

#### **Adding a New API Endpoint**
1. Add endpoint function in appropriate file (`src/kelpie_carbon_v1/api/`)
2. Add Pydantic models for request/response
3. Write unit tests
4. Update API documentation
5. Test manually with Swagger UI

#### **Adding a New Frontend Feature**
1. Update HTML if needed
2. Add JavaScript functionality
3. Update CSS styling
4. Test in multiple browsers
5. Add to user documentation

#### **Adding a New Analysis Feature**
1. Implement core algorithm
2. Add comprehensive tests
3. Integrate with existing pipeline
4. Update documentation
5. Add example usage

---

## **Thank You!**

Thank you for contributing to Kelpie Carbon v1! Your contributions help advance ocean conservation and blue carbon research. Every contribution, no matter how small, makes a difference.

## **Questions?**

If you have questions about contributing, please:
1. Check this guide first
2. Search existing issues and discussions
3. Create a new discussion or issue
4. Tag maintainers if urgent

Happy coding! 🌊 