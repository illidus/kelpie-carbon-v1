# 🤖 New Agent Quick Start Guide

**Date**: January 9, 2025
**Purpose**: Rapid onboarding for AI agents working on Kelpie Carbon v1
**Target**: Claude, Cursor, and other AI coding assistants

This guide gets you productive immediately while following project standards.

---

## 🚀 **5-Minute Setup**

### **Step 1: Understand the Project**
**Kelpie Carbon v1** is a satellite imagery processing system with:
- 🛰️ **Multi-source satellite data integration** (Landsat, Sentinel-2, Planet Labs)
- 🌾 **Carbon sequestration analysis** for agricultural fields
- 📊 **Vegetation health monitoring** with advanced indices
- 🎯 **Field-level precision** analytics

### **Step 2: Read the Mandatory Structure**
**CRITICAL**: You MUST follow the [STANDARDIZATION_GUIDE.md](./STANDARDIZATION_GUIDE.md) - it prevents breaking the architecture.

**Quick Structure Overview:**
```
docs/
├── README.md                     # 🚪 START HERE
├── [Core docs]                   # User guides, API docs
├── research/                     # 🔬 Research documents only
└── implementation/               # 📋 Task summaries only

tests/
├── unit/                         # ⚡ Fast, isolated tests
├── integration/                  # 🔗 Component interaction
├── e2e/                         # 🌐 Full workflow tests
└── performance/                  # ⚡ Performance benchmarks
```

### **Step 3: Essential Files to Review**
1. 📖 **[docs/README.md](./README.md)** - Navigation hub
2. 📋 **[CURRENT_TASK_LIST.md](./CURRENT_TASK_LIST.md)** - YOUR WORK PRIORITIES
3. 🏗️ **[ARCHITECTURE.md](./ARCHITECTURE.md)** - System design
4. 🔧 **[API_REFERENCE.md](./API_REFERENCE.md)** - Available endpoints
5. 📏 **[STANDARDIZATION_GUIDE.md](./STANDARDIZATION_GUIDE.md)** - MANDATORY rules

---

## ⚡ **Immediate Action Guidelines**

### **Before Making ANY Changes**
```bash
# 1. Verify current test status
poetry run pytest

# 2. Check code quality
poetry run black --check src/ tests/
poetry run mypy src/

# 3. Review recent changes
git log --oneline -10
```

### **File Placement Decision Tree**
```
New file needed?
├── 📚 Documentation?
│   ├── User guide/API docs → docs/ (root)
│   ├── Research/validation → docs/research/
│   └── Task summary → docs/implementation/
└── 🧪 Test file?
    ├── Single function test → tests/unit/
    ├── Component integration → tests/integration/
    ├── Full workflow → tests/e2e/
    └── Performance measure → tests/performance/
```

---

## 📋 **Task List Management (CRITICAL)**

### **Primary Task List: YOUR SINGLE SOURCE OF TRUTH**
**📋 `docs/CURRENT_TASK_LIST.md`** contains ALL active tasks. This is where you:
- ✅ **Find your work priorities** (HIGH → MEDIUM → LOW)
- ✅ **Update task progress** as you complete work
- ✅ **Add new tasks** when issues arise
- ✅ **Move completed tasks** to "Recently Completed" section

### **Task Management Rules**
```
BEFORE starting ANY work:
1. 📋 Check CURRENT_TASK_LIST.md for priorities
2. 🔍 Find the appropriate task to work on
3. 📝 Update task status to "IN PROGRESS"

AFTER completing work:
1. ✅ Mark sub-tasks as completed
2. 📊 Update progress metrics
3. 🎯 Move to "Recently Completed" when done
4. 📝 Create implementation summary if significant
```

### **Adding New Tasks (When Issues Arise)**
When you discover bugs, optimizations, or new requirements:

1. ✅ **Assess Priority**: HIGH/MEDIUM/LOW based on project impact
2. ✅ **Add to Primary List**: Place in appropriate priority section
3. ✅ **Use Standard Format**: Follow task template structure
4. ✅ **Note Dependencies**: Document prerequisites and blockers
5. ✅ **Link Details**: Reference detailed task lists if needed

---

## 🎯 **Common Tasks & Patterns**

### **Adding New Features**
1. ✅ **Check Task List**: Ensure feature is in `CURRENT_TASK_LIST.md`
2. ✅ **Code Changes**: Implement in `src/`
3. ✅ **Unit Tests**: Add to `tests/unit/`
4. ✅ **Integration Tests**: Add to `tests/integration/` if needed
5. ✅ **Update API Docs**: Modify `API_REFERENCE.md`
6. ✅ **Implementation Summary**: Create in `docs/implementation/`
7. ✅ **Update Task List**: Mark task completed

### **Bug Fixes**
1. ✅ **Add to Task List**: If not already there, add bug fix task
2. ✅ **Reproduce**: Write failing test first
3. ✅ **Fix Code**: Minimal change to pass test
4. ✅ **Verify**: All tests pass
5. ✅ **Document**: Update `docs/implementation/` with summary
6. ✅ **Update Task List**: Mark bug fix completed

### **Research & Analysis**
1. ✅ **Add Research Task**: Add to task list if significant research needed
2. ✅ **Research Docs**: Place in `docs/research/`
3. ✅ **Validation Tests**: Use `tests/performance/` for benchmarks
4. ✅ **Update Research README**: Add entry to `docs/research/README.md`
5. ✅ **Update Task List**: Mark research task completed

---

## 🚨 **Critical Do's and Don'ts**

### **❌ NEVER Do This**
- ❌ Start work without checking `CURRENT_TASK_LIST.md`
- ❌ Put implementation summaries in `docs/` root
- ❌ Put research docs in `docs/` root
- ❌ Put unit tests with external dependencies
- ❌ Create files without updating relevant READMEs
- ❌ Skip the standardization guide requirements
- ❌ Add tasks to random files instead of the primary task list

### **✅ ALWAYS Do This**
- ✅ Check and update `CURRENT_TASK_LIST.md` before and after work
- ✅ Run tests before and after changes
- ✅ Update appropriate README files
- ✅ Follow naming conventions exactly
- ✅ Add cross-references to related docs
- ✅ Use the provided templates
- ✅ Add new tasks to primary task list when issues arise

---

## 📋 **Quick Reference Templates**

### **Implementation Summary** (use for any completed work)
```markdown
# [Feature] Implementation Summary

**Date**: [YYYY-MM-DD]
**Status**: COMPLETED
**Type**: [Feature/Bug Fix/Optimization]

## 🎯 Objective
[What was implemented]

## ✅ Completed Tasks
- [x] Task 1
- [x] Task 2

## 📊 Results
[Performance metrics, improvements]

## 🧪 Testing
**Test Results**: [Pass/Fail counts]
**Categories**: [unit/integration/e2e/performance]

## 🔗 Related Documentation
- [Links to updated docs]
```

### **Research Document** (for technical analysis)
```markdown
# [Topic] Research Summary

**Date**: [YYYY-MM-DD]
**Type**: [Validation/Analysis/Study]
**Status**: COMPLETED

## 🔬 Objective
[What was studied]

## 📊 Methodology
[How research was conducted]

## 📈 Results
[Findings and conclusions]

## 🔗 Implementation Impact
[How this affects the codebase]
```

---

## 🔧 **Development Environment**

### **Core Technologies**
- **Language**: Python 3.9+
- **Framework**: FastAPI for APIs
- **Testing**: pytest with comprehensive coverage
- **Dependency Management**: Poetry
- **Code Quality**: Black, MyPy, Ruff

### **Key Dependencies**
```python
# Satellite data processing
rasterio, geopandas, xarray

# Machine learning & analysis
scikit-learn, numpy, pandas

# Web framework
fastapi, uvicorn

# Testing
pytest, pytest-asyncio, httpx
```

### **Development Commands**
```bash
# Install dependencies
poetry install

# Run all tests
poetry run pytest

# Run specific test category
poetry run pytest tests/unit/
poetry run pytest tests/integration/

# Code formatting
poetry run black src/ tests/

# Type checking
poetry run mypy src/

# Start development server
poetry run uvicorn src.main:app --reload
```

---

## 📊 **Current System Status**

### **Test Coverage**
- ✅ **205 tests passing** across all categories
- ✅ **Unit tests**: Fast, isolated component testing
- ✅ **Integration tests**: Real satellite data processing
- ✅ **E2E tests**: Complete workflow validation
- ✅ **Performance tests**: Optimization verification

### **Key Capabilities**
- 🛰️ **Multi-satellite integration**: Landsat 8, Sentinel-2, Planet Labs
- 📊 **Advanced indices**: NDVI, EVI, SAVI, Red Edge NDVI
- 🌾 **Carbon analysis**: Sequestration estimation algorithms
- ⚡ **Performance**: Optimized caching and processing
- 🎯 **Precision**: Field-level accuracy with polygon support

---

## 🎯 **Success Checklist**

Before finishing any task:
- [ ] ✅ Task list updated with progress and completion status
- [ ] ✅ All tests pass (`poetry run pytest`)
- [ ] ✅ Code is formatted (`poetry run black --check`)
- [ ] ✅ Types are valid (`poetry run mypy src/`)
- [ ] ✅ Files are in correct directories
- [ ] ✅ READMEs are updated
- [ ] ✅ Implementation summary created (if applicable)
- [ ] ✅ Cross-references added to related docs
- [ ] ✅ New tasks added to primary list if issues discovered

---

## 🤝 **Getting Help**

### **Documentation Navigation**
- 📖 **User Questions**: Check `USER_GUIDE.md`
- 🏗️ **Architecture Questions**: Check `ARCHITECTURE.md`
- 🔧 **API Questions**: Check `API_REFERENCE.md`
- 📏 **Structure Questions**: Check `STANDARDIZATION_GUIDE.md`

### **Code Investigation**
- 🔍 **Find functions**: Use semantic search on codebase
- 🧪 **Understand testing**: Check `tests/README.md`
- 📊 **Performance data**: Check `tests/performance/`
- 🔬 **Research context**: Check `docs/research/`

---

**Welcome to Kelpie Carbon v1! Following this guide ensures you contribute effectively while maintaining the project's careful organization. Every file placement and naming decision supports long-term maintainability.**

**Next Steps**: Read the [STANDARDIZATION_GUIDE.md](./STANDARDIZATION_GUIDE.md) thoroughly, then dive into [ARCHITECTURE.md](./ARCHITECTURE.md) to understand the system design.
