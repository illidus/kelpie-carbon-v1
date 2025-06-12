# 📏 Kelpie Carbon v1 - Standardization Guide

**Date**: January 9, 2025
**Purpose**: Ensure consistent project organization and prevent structural drift
**Audience**: AI Agents, Developers, Contributors

This document establishes **mandatory standards** for maintaining the project's organizational structure. **Deviation from these standards will break the carefully designed architecture.**

---

## 🎯 **Purpose & Scope**

### **Why This Guide Exists**
The Kelpie Carbon v1 project underwent comprehensive reorganization in January 2025 to:
- ✅ Improve navigation and accessibility
- ✅ Categorize documentation by purpose
- ✅ Organize tests by type for efficient development
- ✅ Create sustainable patterns for future growth

### **Who Must Follow This**
- 🤖 **AI Agents** (Claude, Cursor, etc.)
- 👨‍💻 **Human Developers**
- 🤝 **Contributors** (internal and external)
- 📋 **Project Maintainers**

---

## 🏗️ **Mandatory Directory Structure**

### **Documentation Structure (FIXED)**
```
docs/
├── README.md                     # 🚪 ENTRY POINT - Navigation hub
├── [Core Documentation Files]    # User guides, API docs, architecture
│
├── research/                     # 🔬 RESEARCH ONLY
│   ├── README.md                 # Research navigation
│   └── [Research documents]      # Validation, analysis, studies
│
└── implementation/               # 📋 HISTORY ONLY
    ├── README.md                 # Implementation navigation
    └── [Implementation summaries] # Task records, status tracking
```

### **Test Structure (FIXED)**
```
tests/
├── README.md                     # 📖 Test guide
├── conftest.py                   # Shared configuration
│
├── unit/                         # ⚡ Fast, isolated tests
├── integration/                  # 🔗 Component interaction tests
├── e2e/                         # 🌐 End-to-end workflow tests
└── performance/                  # ⚡ Performance & optimization tests
```

---

## 📋 **File Placement Rules (MANDATORY)**

### **Core Documentation** (`docs/` root)
**PLACE HERE:**
- ✅ User guides and tutorials
- ✅ API reference documentation
- ✅ Architecture and design docs
- ✅ Deployment and setup guides
- ✅ Developer onboarding materials

**EXAMPLES:**
- `USER_GUIDE.md`, `API_REFERENCE.md`
- `ARCHITECTURE.md`, `DEPLOYMENT_GUIDE.md`
- `DEVELOPER_ONBOARDING.md`

### **Research Documentation** (`docs/research/`)
**PLACE HERE:**
- ✅ Scientific validation frameworks
- ✅ Algorithm research and analysis
- ✅ Data validation studies
- ✅ Technical specifications for research

**EXAMPLES:**
- `VALIDATION_DATA_FRAMEWORK.md`
- `RED_EDGE_ENHANCEMENT_SPEC.md`
- `ALGORITHM_COMPARISON_STUDY.md`

### **Implementation History** (`docs/implementation/`)
**PLACE HERE:**
- ✅ Task completion summaries
- ✅ Phase implementation records
- ✅ Optimization and performance tracking
- ✅ Bug fix documentation
- ✅ Status and progress reports

**EXAMPLES:**
- `PHASE_5_COMPLETION_SUMMARY.md`
- `OPTIMIZATION_CACHING_IMPLEMENTATION.md`
- `BUGFIX_IMAGERY_RESOLUTION_SUMMARY.md`

---

## 🧪 **Test Categorization Rules (MANDATORY)**

### **Unit Tests** (`tests/unit/`)
```python
# CRITERIA:
✅ Fast execution (< 1 second)
✅ No external dependencies
✅ Test single functions/classes
✅ Mockable dependencies

# EXAMPLES:
- test_api.py (API endpoint logic)
- test_models.py (data models)
- test_utils.py (utility functions)
```

### **Integration Tests** (`tests/integration/`)
```python
# CRITERIA:
✅ Test component interactions
✅ May use external APIs
✅ Test real data flows
✅ Verify service integrations

# EXAMPLES:
- test_satellite_integration.py
- test_database_integration.py
- test_real_satellite_data.py
```

### **End-to-End Tests** (`tests/e2e/`)
```python
# CRITERIA:
✅ Test complete user workflows
✅ Full system integration
✅ Simulate real user scenarios
✅ Test entire pipeline

# EXAMPLES:
- test_integration_comprehensive.py
- test_user_workflow_complete.py
```

### **Performance Tests** (`tests/performance/`)
```python
# CRITERIA:
✅ Measure performance metrics
✅ Test optimization effectiveness
✅ Benchmark improvements
✅ Resource usage validation

# EXAMPLES:
- test_optimization.py
- test_phase5_performance.py
- test_caching_effectiveness.py
```

---

## 📝 **Naming Conventions (ENFORCED)**

### **Documentation Files**
```bash
# Core documentation (docs/ root)
USER_GUIDE.md
API_REFERENCE.md
ARCHITECTURE.md
DEPLOYMENT_GUIDE.md

# Research documentation (docs/research/)
VALIDATION_DATA_FRAMEWORK.md
[ALGORITHM]_ENHANCEMENT_SPEC.md
[FEATURE]_RESEARCH_SUMMARY.md

# Implementation summaries (docs/implementation/)
PHASE_[N]_COMPLETION_SUMMARY.md
FEATURE_[NAME]_IMPLEMENTATION_SUMMARY.md
OPTIMIZATION_[ASPECT]_COMPLETION_SUMMARY.md
BUGFIX_[ISSUE]_RESOLUTION_SUMMARY.md
```

### **Test Files**
```bash
# Unit tests
test_[module_name].py

# Integration tests
test_[feature]_integration.py

# End-to-end tests
test_[workflow]_comprehensive.py

# Performance tests
test_[aspect]_performance.py
```

---

## ✅ **Required Actions Checklist**

### **When Adding ANY Documentation**
1. ✅ **Determine Category**: Core, Research, or Implementation?
2. ✅ **Check Directory**: Is there a similar document already?
3. ✅ **Update README**: Add entry to appropriate directory README
4. ✅ **Add Cross-References**: Link from related documents
5. ✅ **Follow Naming**: Use established naming patterns

### **When Adding ANY Test**
1. ✅ **Categorize Correctly**: Unit, Integration, E2E, or Performance?
2. ✅ **Place in Right Directory**: Follow test categorization rules
3. ✅ **Add `__init__.py`**: If creating new test subdirectory
4. ✅ **Update Test README**: Document new test category if needed
5. ✅ **Verify Execution**: Ensure tests run in their category

### **When Creating Implementation Summaries**
1. ✅ **Use Template**: Follow the established summary template
2. ✅ **Place in Implementation**: MUST go in `docs/implementation/`
3. ✅ **Update Implementation README**: Add to appropriate section
4. ✅ **Link Related Docs**: Reference relevant documentation
5. ✅ **Include Test Results**: Document testing outcomes

### **When Managing Task Lists**
1. ✅ **Use Primary Task List**: ALL active tasks go in `docs/CURRENT_TASK_LIST.md`
2. ✅ **Update Status**: Mark completed tasks, update progress
3. ✅ **Add New Tasks**: Append to appropriate priority section
4. ✅ **Cross-Reference**: Link to detailed task lists in appropriate directories
5. ✅ **Maintain Priority Order**: Keep HIGH → MEDIUM → LOW structure

---

## 🚨 **Critical Violations (DO NOT COMMIT)**

### **Directory Violations**
❌ Implementation summaries in `docs/` root
❌ Research documents in `docs/` root
❌ Core documentation in subdirectories
❌ Tests in wrong categories
❌ Missing `__init__.py` in test directories

### **Documentation Violations**
❌ Broken cross-references
❌ Outdated README files
❌ Missing entries in directory READMEs
❌ Inconsistent naming patterns
❌ Duplicate documentation

### **Test Violations**
❌ Unit tests with external dependencies
❌ Integration tests in unit directory
❌ Performance tests without metrics
❌ Missing test categorization
❌ Broken test imports

---

## 📋 **Templates (MANDATORY USE)**

### **Implementation Summary Template**
```markdown
# [Feature/Task] Implementation Summary

**Date**: [YYYY-MM-DD]
**Status**: [COMPLETED/IN_PROGRESS/BLOCKED]
**Type**: [Feature/Bug Fix/Optimization/Research]

## 🎯 Objective
[Clear description of what was implemented/changed]

## ✅ Completed Tasks
- [ ] Task 1: [Description]
- [ ] Task 2: [Description]
- [ ] Task 3: [Description]

## 📊 Results
[Quantitative results, performance improvements, metrics]

## 🧪 Testing
**Test Results**: [Pass/Fail counts, coverage info]
**Test Categories**: [Which test categories were affected]

## 🔗 Related Documentation
- [Link to User Guide updates]
- [Link to API Reference changes]
- [Link to Architecture modifications]

## 📝 Notes
[Any additional context, challenges overcome, lessons learned]
```

### **Research Document Template**
```markdown
# [Research Topic] - [Brief Description]

**Date**: [YYYY-MM-DD]
**Research Type**: [Validation/Analysis/Integration/Study]
**Status**: [COMPLETED/ONGOING/PAUSED]

## 🔬 Research Objective
[What was studied, validated, or analyzed]

## 📊 Methodology
[How the research was conducted, tools used, approach taken]

## 📈 Results
[Findings, data, conclusions, visualizations]

## 🔗 Implementation Impact
[How this research affects the codebase, what changes it enables]

## 📚 References
[Scientific papers, external resources, related research]
```

### **Task List Management Template**
```markdown
### **Task [ID]: [Task Name]** [Emoji]
**Status**: [🟡 IN_PROGRESS/🟢 COMPLETED/🔴 BLOCKED/⚪ NOT_STARTED]
**Priority**: [HIGH/MEDIUM/LOW]
**Estimated Duration**: [Time estimate]
**Prerequisite**: [Dependencies]

#### **Objective**
[Clear description of task goals]

#### **Sub-tasks**
- [ ] **[ID.1]**: [Specific actionable item]
- [ ] **[ID.2]**: [Specific actionable item]
- [ ] **[ID.3]**: [Specific actionable item]

#### **Deliverables**
- [ ] [Specific output 1]
- [ ] [Specific output 2]

#### **Success Metrics**
- [Measurable criteria for completion]
```

---

## 📋 **Task List Management (MANDATORY)**

### **Primary Task List** (`docs/CURRENT_TASK_LIST.md`)
**This is the SINGLE SOURCE OF TRUTH for all active tasks.**

#### **Structure Requirements**
```markdown
# 📋 Current Task List - Kelpie Carbon v1

## 🚨 **IMMEDIATE HIGH PRIORITY TASKS**
[Urgent tasks that must be completed first]

## 🎯 **MEDIUM PRIORITY TASKS**
[Important tasks for next phase]

## 📚 **LOW PRIORITY TASKS**
[Future enhancements and optimizations]

## ✅ **RECENTLY COMPLETED TASKS**
[Completed tasks for reference - move here when done]
```

#### **Task Management Rules**
1. ✅ **Single Source**: ALL active tasks MUST be in `docs/CURRENT_TASK_LIST.md`
2. ✅ **Priority Order**: Maintain HIGH → MEDIUM → LOW structure
3. ✅ **Status Updates**: Update progress regularly with status indicators
4. ✅ **Completion Movement**: Move completed tasks to "Recently Completed" section
5. ✅ **Cross-References**: Link to detailed task lists when they exist elsewhere

### **Detailed Task Lists** (Optional Supplements)
When tasks are complex enough to warrant detailed breakdowns:

#### **Placement Rules**
- **Research Tasks**: Detailed lists can go in `docs/research/` (e.g., `SKEMA_INTEGRATION_TASK_LIST.md`)
- **Implementation Tasks**: Detailed lists can go in `docs/implementation/`
- **Validation Tasks**: Detailed lists can go in `docs/` root for major validation efforts

#### **Integration Requirements**
1. ✅ **Reference in Primary**: Always link from `CURRENT_TASK_LIST.md` to detailed lists
2. ✅ **Update Both**: Keep both primary and detailed lists synchronized
3. ✅ **Clear Hierarchy**: Detailed lists supplement, never replace primary list

### **Task Addition Process**
When adding new tasks (agents and humans):

1. ✅ **Assess Priority**: HIGH/MEDIUM/LOW based on project impact
2. ✅ **Place in Primary List**: Add to appropriate priority section
3. ✅ **Use Template**: Follow task template structure
4. ✅ **Add Dependencies**: Note prerequisites and blockers
5. ✅ **Link Details**: Reference detailed task lists if they exist

### **Task Update Process**
When updating task progress:

1. ✅ **Update Status**: Change status indicators (🟡→🟢, etc.)
2. ✅ **Mark Sub-tasks**: Check off completed sub-tasks
3. ✅ **Update Metrics**: Report progress in quantifiable terms
4. ✅ **Move When Done**: Move completed tasks to "Recently Completed"
5. ✅ **Update Cross-References**: Synchronize with detailed lists

---

## 🔄 **README Maintenance (CRITICAL)**

### **Directory READMEs Must Be Updated**
Every time you add a file to any directory, you MUST update the appropriate README:

1. **`docs/README.md`**: For major additions to core documentation
2. **`docs/research/README.md`**: For all research documents
3. **`docs/implementation/README.md`**: For all implementation summaries
4. **`tests/README.md`**: For new test categories or significant changes

### **Cross-Reference Updates**
When adding documentation:
1. ✅ Link FROM related documents TO your new document
2. ✅ Link FROM your new document TO related documents
3. ✅ Update navigation paths in main README
4. ✅ Verify all links work after file movements

---

## 🎯 **Quality Assurance**

### **Before Committing Changes**
Run this checklist:

```bash
# 1. Verify all tests pass
poetry run pytest

# 2. Check code formatting
poetry run black --check src/ tests/

# 3. Verify type checking
poetry run mypy src/

# 4. Check documentation links
# [Manual verification of cross-references]

# 5. Confirm file placement
# [Verify files are in correct directories]
```

### **Structural Integrity Verification**
1. ✅ **Documentation Navigation**: Can users find what they need?
2. ✅ **Test Execution**: Do test categories run independently?
3. ✅ **Cross-References**: Do all internal links work?
4. ✅ **README Currency**: Are directory READMEs up to date?

---

## 📊 **Success Metrics**

### **Organizational Health**
- ✅ All documentation in correct directories
- ✅ All tests in appropriate categories
- ✅ All README files current and accurate
- ✅ All cross-references working
- ✅ Clear navigation paths for all user types

### **Maintenance Efficiency**
- ✅ New contributors can navigate quickly
- ✅ AI agents follow structure consistently
- ✅ Test execution is fast and targeted
- ✅ Documentation updates are straightforward

---

## 🚀 **Future-Proofing**

### **Scalability Considerations**
As the project grows:
1. ✅ **Maintain Categories**: Don't create new top-level directories
2. ✅ **Use Subdirectories**: Organize within existing structure
3. ✅ **Update Templates**: Evolve templates while maintaining structure
4. ✅ **Document Changes**: Any structural evolution must be documented

### **Structure Evolution**
If structural changes become necessary:
1. ✅ **Document Rationale**: Why is change needed?
2. ✅ **Migration Plan**: How to move existing content?
3. ✅ **Update All Guides**: Agent guide, standardization guide, READMEs
4. ✅ **Verify Integrity**: Ensure no broken references

---

## 📞 **Enforcement & Support**

### **For AI Agents**
- 📖 **Read This Guide First** before making any structural changes
- 🔍 **Check Existing Structure** before creating new files
- ✅ **Follow Templates** for consistent documentation
- 🔗 **Update Cross-References** when adding content

### **For Human Developers**
- 📋 **Review During Code Review** to catch structural violations
- 🎯 **Use PR Templates** that include structural checklist
- 📝 **Document Deviations** if emergency changes are needed
- 🤝 **Educate Contributors** on these standards

---

**Remember**: This structure was designed for long-term maintainability. Every deviation makes the project harder to navigate and maintain. Following these standards ensures the project remains accessible and manageable as it grows.

**Verification**: These standards maintain the current success metrics:
- ✅ 205 tests passing across all categories
- ✅ Clear documentation navigation
- ✅ Sustainable organizational patterns
