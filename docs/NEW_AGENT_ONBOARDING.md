# 🤖 New AI Agent Onboarding Guide

**Date**: January 9, 2025  
**Purpose**: Get new AI agents productive immediately with the established project structure  
**Critical**: Follow this guide EXACTLY to maintain project integrity

---

## 🚨 **CRITICAL FIRST STEPS - READ BEFORE DOING ANYTHING**

### **1. Read These Documents FIRST (In Order)**
1. **[📏 STANDARDIZATION_GUIDE.md](STANDARDIZATION_GUIDE.md)** - **MANDATORY** organizational standards
2. **[🤖 agent-guide.md](agent-guide.md)** - Your comprehensive instructions and workflow
3. **[📚 README.md](README.md)** - Project navigation hub
4. **[🏗️ ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture overview

### **2. Project Status Check**
- ✅ **Status**: Production-ready system (All 5 phases complete)
- ✅ **Test Suite**: 205 tests passing, 7 skipped
- ✅ **Documentation**: Fully organized and structured
- ✅ **Test Organization**: Categorized by type (unit, integration, e2e, performance)

---

## 📋 **Essential Project Context**

### **What This Project Is**
- **Name**: Kelpie Carbon v1
- **Purpose**: Real-time kelp forest carbon sequestration assessment using satellite imagery
- **Tech Stack**: FastAPI + Python 3.12, Leaflet.js, Sentinel-2 data processing
- **Current Status**: Production-ready with comprehensive test coverage

### **Key Capabilities** ✅
- Real-time Sentinel-2 satellite data processing
- Interactive web interface with progressive loading
- Spectral index calculations (NDVI, FAI, NDRE) 
- Machine learning kelp detection
- Carbon sequestration estimation

### **Current Focus Areas**
- **Next Priority**: SKEMA integration for enhanced kelp detection
- **Research Phase**: Validating SKEMA formulas against known kelp farm locations
- **Quality Focus**: Maintaining 100% test coverage and clean architecture

---

## 🎯 **Your Immediate Actions**

### **Step 1: Verify Project Structure Understanding**
Check that you understand the **MANDATORY** structure:

```
docs/
├── README.md                          # 🚪 Navigation hub - START HERE
├── STANDARDIZATION_GUIDE.md           # 📏 MANDATORY standards
├── agent-guide.md                     # 🤖 Your primary guide
├── [Core Documentation]               # User guides, API docs, architecture
├── research/                          # 🔬 Research & validation docs ONLY
└── implementation/                    # 📋 Implementation history ONLY

tests/
├── unit/                              # ⚡ Fast, isolated tests
├── integration/                       # 🔗 Component interaction tests  
├── e2e/                              # 🌐 End-to-end workflow tests
└── performance/                       # ⚡ Performance tests
```

### **Step 2: Review Current Task List**
Read the current task list at:
- **[📋 CURRENT_TASK_LIST.md](CURRENT_TASK_LIST.md)** - Your work priorities
- **[🔬 research/SKEMA_INTEGRATION_TASK_LIST.md](research/SKEMA_INTEGRATION_TASK_LIST.md)** - SKEMA research tasks

### **Step 3: Verify Development Environment**
```bash
# Ensure all tests pass before starting work
poetry run pytest

# Check code quality
poetry run black --check src/ tests/
poetry run mypy src/
poetry run flake8 src/ tests/
```

---

## 🔄 **Development Workflow**

### **Before Making ANY Changes**
1. ✅ **Read standardization guide** - Understand mandatory rules
2. ✅ **Check task list** - Ensure your work aligns with priorities  
3. ✅ **Verify tests pass** - Start with clean baseline
4. ✅ **Review architecture** - Understand component interactions

### **When Working on Tasks**
1. **Identify Category**: Is this core functionality, research, or implementation history?
2. **Update Tests**: Add/modify tests in appropriate category (unit/integration/e2e/performance)
3. **Document Work**: Use templates from STANDARDIZATION_GUIDE.md
4. **Maintain Quality**: Ensure all tests pass before committing

### **When Creating Documentation**
1. **Determine Placement**: Core (docs/), Research (docs/research/), or Implementation (docs/implementation/)
2. **Use Templates**: Follow mandatory templates in STANDARDIZATION_GUIDE.md
3. **Update READMEs**: Add entries to appropriate directory README
4. **Cross-Reference**: Link from related documents

---

## 🎯 **Current Priority Tasks**

### **Immediate High Priority**
1. **Fix Pre-commit Hooks** - Address current linting/type checking issues
2. **SKEMA Integration Phase 3** - Begin model validation with known kelp farm locations
3. **Enhanced Testing** - Validate system against SKEMA-recognized kelp farms

### **SKEMA Integration Focus**
- **Goal**: Integrate SKEMA formulas and validate against known kelp farm coordinates
- **Critical**: Test lat/lng coordinates that SKEMA recognizes as true kelp farms
- **Success Metric**: Our indices successfully detect SKEMA-validated kelp locations

### **Quality Maintenance**
- **Test Coverage**: Maintain 100% passing tests
- **Documentation**: Keep all docs current and cross-referenced
- **Code Quality**: Fix any linting/type checking issues

---

## 📋 **Task Management Standards**

### **When You Complete Tasks**
1. **Create Implementation Summary**: Use template in docs/implementation/
2. **Update Task Lists**: Mark completed items, add new discoveries
3. **Document Results**: Include metrics, test results, lessons learned
4. **Cross-Reference**: Link to related documentation updates

### **Task Documentation Template**
```markdown
# [TASK] Implementation Summary

**Date**: [YYYY-MM-DD]
**Status**: COMPLETED
**Type**: [Feature/Research/Optimization/Bug Fix]

## 🎯 Objective
[What was accomplished]

## ✅ Completed Work
- [Specific item 1]
- [Specific item 2]

## 📊 Results
[Metrics, improvements, discoveries]

## 🧪 Testing
**Test Results**: [Pass/Fail counts]
**New Tests**: [Any new tests added]

## 🔗 Related Documentation
- [Links to updated docs]

## 📝 Next Steps
[What this enables for future work]
```

---

## 🚨 **Critical Don'ts - NEVER Do These**

### **Structure Violations**
- ❌ **NEVER** put implementation summaries in `docs/` root
- ❌ **NEVER** put research docs in `docs/` root  
- ❌ **NEVER** mix test types in wrong directories
- ❌ **NEVER** skip updating README files when adding documentation
- ❌ **NEVER** break cross-references without fixing them

### **Development Violations**
- ❌ **NEVER** commit code without running tests
- ❌ **NEVER** ignore the standardization guide rules
- ❌ **NEVER** create documentation without using templates
- ❌ **NEVER** work on tasks not in the current task list

---

## ✅ **Success Checklist**

Before considering any work session complete:

### **Code Quality**
- [ ] All tests passing (pytest shows 205+ passed)
- [ ] No linting errors (flake8 passes)  
- [ ] No type errors (mypy passes)
- [ ] Code formatted (black passes)

### **Documentation**
- [ ] Any new docs placed in correct directory
- [ ] README files updated for new content
- [ ] Cross-references working
- [ ] Templates followed

### **Task Management** 
- [ ] Task list updated with progress
- [ ] Implementation summary created (if task completed)
- [ ] Next steps documented
- [ ] Related documentation linked

---

## 🎯 **Current SKEMA Integration Status**

### **Research Phase Complete** ✅
- Task 1: SKEMA framework research - COMPLETED
- NDRE implementation - COMPLETED  
- Red-edge enhancement specification - COMPLETED

### **Next Critical Phase** 🎯
- **Task 3**: Model validation with SKEMA-recognized kelp farm locations
- **Goal**: Prove our indices detect kelp at coordinates SKEMA validates as true kelp farms
- **Success Metric**: High correlation between our detection and SKEMA ground truth

### **Key SKEMA Integration Components**
- Enhanced NDRE (Normalized Difference Red Edge) processing
- Multi-spectral index calculations (FAI, NDVI, NDRE)
- Deep learning model integration
- Temporal validation across environmental conditions

---

## 📞 **Getting Help**

### **Documentation Resources**
1. **Confused about structure?** → Read [STANDARDIZATION_GUIDE.md](STANDARDIZATION_GUIDE.md)
2. **Need development guidance?** → Read [agent-guide.md](agent-guide.md)  
3. **Want to understand architecture?** → Read [ARCHITECTURE.md](ARCHITECTURE.md)
4. **Looking for specific APIs?** → Read [API_REFERENCE.md](API_REFERENCE.md)

### **Common Issues & Solutions**
- **Tests failing?** → Check test categorization and dependencies
- **Documentation placement unclear?** → Use the file placement rules in standardization guide
- **Pre-commit hooks failing?** → This is a current known issue, add to task list
- **SKEMA integration questions?** → Check research/ directory for specifications

---

## 🏁 **Ready to Start**

Once you've read all required documents and understand the structure:

1. **Check current task list** → [CURRENT_TASK_LIST.md](CURRENT_TASK_LIST.md)
2. **Pick highest priority task** → Focus on SKEMA integration or pre-commit fixes
3. **Follow development workflow** → Test, code, document, test again
4. **Maintain standards** → Use templates, update READMEs, cross-reference

**Remember**: This project has been carefully organized for maintainability. Following the established patterns ensures continued success and makes your work valuable for future agents and developers.

**Goal**: Integrate SKEMA methodology and validate our kelp detection against known kelp farm locations to prove system effectiveness.

---

**Last Updated**: January 9, 2025  
**Next Review**: After each major task completion  
**Purpose**: Ensure new agents are immediately productive while maintaining project integrity 