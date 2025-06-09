# 🤖 AI Agent Guide - Kelpie Carbon v1

**Last Updated**: January 9, 2025  
**Structure Version**: 2.0 (Post-Reorganization)

This guide provides comprehensive instructions for AI agents working on the Kelpie Carbon v1 project. **Follow this guide carefully to maintain the established organizational structure.**

---

## 📋 **Project Overview**

### **Project Details**
- **Name**: Kelpie Carbon v1
- **Purpose**: Kelp forest carbon sequestration assessment using satellite imagery
- **Tech Stack**: FastAPI + Python 3.12, Leaflet.js, Sentinel-2 data
- **Package Path**: `src/kelpie_carbon_v1/`
- **CLI Entry**: `kelpie-carbon-v1`
- **Status**: Production-ready (All 5 phases complete)

### **Current Capabilities** ✅
- ✅ Real-time Sentinel-2 satellite data processing
- ✅ Interactive web interface with progressive loading
- ✅ Spectral index calculations (NDVI, FAI, NDRE)
- ✅ Machine learning kelp detection
- ✅ Carbon sequestration estimation
- ✅ Comprehensive test suite (205 tests)
- ✅ Complete documentation

---

## 🏗️ **Project Structure (CRITICAL - Follow Exactly)**

### **Documentation Organization**
```
docs/
├── README.md                          # 📚 NAVIGATION HUB - Start here
├── PROJECT_SUMMARY.md                 # Project overview
├── USER_GUIDE.md                      # End-user documentation  
├── DEVELOPER_ONBOARDING.md            # Developer setup
├── DEVELOPMENT_GUIDE.md               # Development workflows
├── ARCHITECTURE.md                    # System architecture
├── API_REFERENCE.md                   # API documentation
├── TESTING_GUIDE.md                   # Testing strategies
├── DEPLOYMENT_GUIDE.md                # Production deployment
├── SATELLITE_IMAGERY_FEATURE.md       # Satellite processing
├── web-interface.md                   # Frontend documentation
├── agent-guide.md                     # This file - AI agent guide
│
├── research/                          # 🔬 Research & validation docs
│   ├── README.md                      # Research navigation
│   ├── VALIDATION_DATA_FRAMEWORK.md   # SKEMA validation
│   ├── RED_EDGE_ENHANCEMENT_SPEC.md   # NDRE research
│   ├── SKEMA_INTEGRATION_TASK_LIST.md # Research tasks
│   ├── SKEMA_RESEARCH_SUMMARY.md      # Research findings
│   └── NDRE_IMPLEMENTATION_SUCCESS.md # Implementation results
│
└── implementation/                    # 📋 Historical implementation docs
    ├── README.md                      # Implementation navigation
    ├── [Phase implementation summaries]
    ├── [Task completion records]
    ├── [Optimization documentation]
    └── [Status tracking files]
```

### **Test Organization**
```
tests/
├── README.md                          # 📖 Test suite guide
├── conftest.py                        # Shared test configuration
│
├── unit/                              # ⚡ Unit tests (fast, isolated)
│   ├── __init__.py
│   └── [12 unit test files]
│
├── integration/                       # 🔗 Integration tests (external APIs)
│   ├── __init__.py
│   └── [4 integration test files]
│
├── e2e/                              # 🌐 End-to-end tests (workflows)
│   ├── __init__.py
│   └── [1 comprehensive test file]
│
└── performance/                       # ⚡ Performance tests
    ├── __init__.py
    └── [2 performance test files]
```

---

## 📖 **Documentation Standards (MANDATORY)**

### **File Placement Rules**
1. **Core Documentation**: Place in `docs/` root
   - User guides, API docs, architecture, deployment
   - **Examples**: `USER_GUIDE.md`, `API_REFERENCE.md`

2. **Research Documentation**: Place in `docs/research/`
   - Validation frameworks, scientific specifications
   - **Examples**: New SKEMA research, spectral analysis studies

3. **Implementation History**: Place in `docs/implementation/`
   - Task summaries, optimization records, status updates
   - **Examples**: Future implementation summaries, troubleshooting docs

### **New File Creation Checklist**
Before creating ANY new documentation file:

1. ✅ **Determine Category**: Core, Research, or Implementation?
2. ✅ **Check Existing**: Does similar documentation already exist?
3. ✅ **Update READMEs**: Add entry to appropriate directory README
4. ✅ **Add Cross-References**: Link from related documents
5. ✅ **Follow Naming**: Use clear, descriptive filenames

### **Documentation Templates**

#### **Implementation Summary Template**
```markdown
# [Feature/Task] Implementation Summary

**Date**: [Date]
**Status**: [COMPLETED/IN_PROGRESS/BLOCKED]
**Type**: [Feature/Bug Fix/Optimization/Research]

## 🎯 Objective
[What was implemented/changed]

## ✅ Completed Tasks
- [ ] Task 1
- [ ] Task 2

## 📊 Results
[Metrics, performance improvements, etc.]

## 🔗 Related Documentation
- [Link to related docs]

## 🧪 Testing
[Test results, coverage info]
```

#### **Research Document Template**
```markdown
# [Research Topic] - [Brief Description]

**Date**: [Date]
**Research Type**: [Validation/Analysis/Integration]
**Status**: [COMPLETED/ONGOING]

## 🔬 Research Objective
[What was studied/validated]

## 📊 Methodology
[How research was conducted]

## 📈 Results
[Findings, data, conclusions]

## 🔗 Implementation Impact
[How this affects the codebase]
```

---

## 🧪 **Testing Standards (CRITICAL)**

### **Test Categorization Rules**
1. **Unit Tests** (`tests/unit/`): 
   - Fast, isolated, no external dependencies
   - Test individual functions/classes
   - **Example**: API endpoint logic, data models

2. **Integration Tests** (`tests/integration/`):
   - Test component interactions
   - May use external APIs (satellite data)
   - **Example**: Real satellite data fetching

3. **End-to-End Tests** (`tests/e2e/`):
   - Test complete user workflows
   - Full system integration
   - **Example**: Complete analysis pipeline

4. **Performance Tests** (`tests/performance/`):
   - Test optimization and resource usage
   - **Example**: Caching effectiveness, response times

### **Test File Naming**
- `test_[module_name].py` for unit tests
- `test_[feature]_integration.py` for integration tests
- `test_[workflow]_comprehensive.py` for e2e tests
- `test_[aspect]_performance.py` for performance tests

### **Test Execution Commands**
```bash
# Unit tests (fast feedback)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v

# Performance tests
pytest tests/performance/ -v

# All tests
pytest -v
```

---

## 🛠️ **Development Workflow**

### **Before Making Changes**
1. ✅ **Read Documentation Index**: Start at `docs/README.md`
2. ✅ **Check Architecture**: Review `docs/ARCHITECTURE.md`
3. ✅ **Review Tests**: Check `tests/README.md` for testing strategy
4. ✅ **Understand Structure**: Follow the organization established

### **Code Changes Process**
1. **Identify Impact**: What components are affected?
2. **Update Tests**: Add/modify tests in appropriate category
3. **Update Documentation**: Update relevant docs
4. **Run Tests**: Ensure all tests pass
5. **Update Cross-References**: Fix any broken links

### **Quality Checks**
```bash
# Run all tests
poetry run pytest

# Check code formatting
poetry run black --check src/ tests/

# Type checking
poetry run mypy src/

# Lint checking
poetry run flake8 src/ tests/
```

---

## 📝 **Documentation Maintenance**

### **When Adding New Features**
1. **Update Core Docs**: Modify `API_REFERENCE.md`, `USER_GUIDE.md` as needed
2. **Add Implementation Summary**: Create summary in `docs/implementation/`
3. **Update Architecture**: If system design changes
4. **Cross-Reference**: Link from related documents

### **When Adding Research**
1. **Create Research Doc**: Place in `docs/research/`
2. **Update Research README**: Add entry to `docs/research/README.md`
3. **Link Implementation**: Reference from implementation summaries
4. **Update Main Index**: Add to `docs/README.md` if significant

### **When Completing Tasks**
1. **Create Summary**: Document what was done
2. **Move to Implementation**: Place in `docs/implementation/`
3. **Update Status**: Mark tasks as complete
4. **Archive Temporary Files**: Remove temporary documentation

---

## 🚨 **Critical Rules (DO NOT BREAK)**

### **Structure Preservation**
1. ❌ **NEVER** put implementation summaries in `docs/` root
2. ❌ **NEVER** put research docs in `docs/` root  
3. ❌ **NEVER** mix test types in wrong directories
4. ❌ **NEVER** skip updating READMEs when adding files
5. ❌ **NEVER** break cross-references without fixing them

### **Required Actions**
1. ✅ **ALWAYS** update appropriate README when adding files
2. ✅ **ALWAYS** categorize tests correctly
3. ✅ **ALWAYS** add cross-references to related documents
4. ✅ **ALWAYS** follow established naming conventions
5. ✅ **ALWAYS** test changes before committing

### **Documentation Links**
1. ✅ **ALWAYS** use relative paths for internal links
2. ✅ **ALWAYS** update `docs/README.md` for major additions
3. ✅ **ALWAYS** check links work after moving files
4. ✅ **ALWAYS** maintain directory README files

---

## 🗂️ **Future Implementation Summaries**

### **Storage Location**
All future implementation summaries MUST go in `docs/implementation/` with clear naming:
- `FEATURE_[NAME]_IMPLEMENTATION_SUMMARY.md`
- `BUGFIX_[ISSUE]_RESOLUTION_SUMMARY.md`
- `OPTIMIZATION_[ASPECT]_COMPLETION_SUMMARY.md`
- `RESEARCH_[TOPIC]_INTEGRATION_SUMMARY.md`

### **Required Content**
1. **Date and Status**
2. **Clear Objective**
3. **Tasks Completed**
4. **Results/Metrics**
5. **Test Results**
6. **Related Documentation Links**

### **Directory README Update**
ALWAYS add new summaries to `docs/implementation/README.md` under the appropriate category.

---

## 🔄 **Version Control Best Practices**

### **Commit Strategy**
1. **Small, Focused Commits**: One logical change per commit
2. **Clear Messages**: Use conventional commit format
3. **Test Before Commit**: Ensure tests pass
4. **Document Changes**: Update relevant documentation

### **Conventional Commits**
```bash
feat(api): add new kelp detection endpoint
fix(imagery): resolve RGB composite caching issue
docs(guide): update agent guide with new structure
test(unit): add comprehensive model validation tests
refactor(core): improve satellite data processing efficiency
```

---

## 🎯 **Success Metrics**

### **Code Quality**
- ✅ All tests passing (currently 205 passed, 7 skipped)
- ✅ Clean code following established patterns
- ✅ Proper error handling and logging
- ✅ Type hints and documentation

### **Documentation Quality**
- ✅ Clear, up-to-date documentation
- ✅ Proper file organization
- ✅ Working cross-references
- ✅ Comprehensive implementation tracking

### **Structural Integrity**
- ✅ Files in correct directories
- ✅ Updated README files
- ✅ Maintained test categorization
- ✅ Preserved architectural patterns

---

## 📞 **Getting Help**

### **Documentation Resources**
1. **Start Here**: `docs/README.md` - Navigation hub
2. **Development**: `docs/DEVELOPER_ONBOARDING.md` - Setup guide
3. **Testing**: `tests/README.md` - Test guidance
4. **Architecture**: `docs/ARCHITECTURE.md` - System design

### **Common Issues**
1. **Test Failures**: Check test category and dependencies
2. **Documentation Links**: Verify relative paths are correct
3. **File Placement**: Follow the established directory structure
4. **Integration**: Ensure external APIs are accessible

---

**Remember**: This organizational structure was carefully designed for maintainability. Following these guidelines ensures the codebase remains clean, navigable, and sustainable for future development.

**Last Verification**: All 205 tests passing ✅  
**Structure Status**: Fully implemented and documented ✅