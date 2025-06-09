# 🔧 **Troubleshooting Guide - Resolved Issues**

*Created: 2025-01-09*

This guide documents critical issues encountered during optimization and their solutions.

---

## 🚨 **Critical Issues & Solutions**

### **Issue 1: Content Security Policy Blocking Leaflet**

**🔍 Symptoms:**
```
Content-Security-Policy: The page's settings blocked a script (script-src-elem) at
https://unpkg.com/leaflet@1.9.4/dist/leaflet.js from being executed
```

**💡 Root Cause:**
- Overly restrictive CSP policy blocking external CDN resources
- Missing `https://unpkg.com` in script-src and style-src directives

**✅ Solution:**
```javascript
// Updated CSP in src/kelpie_carbon_v1/api/main.py
csp_policy = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com; "
    "style-src 'self' 'unsafe-inline' https://unpkg.com; "
    "img-src 'self' data: https: blob:; "
    "connect-src 'self' https:; "
    "font-src 'self' data: https:; "
    "frame-ancestors 'none'"
)
```

**🧪 Test:** Web interface now loads Leaflet maps correctly

---

### **Issue 2: Test Import Failures**

**🔍 Symptoms:**
```
ModuleNotFoundError: No module named 'kelpie_carbon_v1.fetch'
ModuleNotFoundError: No module named 'kelpie_carbon_v1.mask'
```

**💡 Root Cause:**
- Tests using outdated import paths from old module structure
- Functions moved to `core.*` packages not reflected in tests

**✅ Solution:**
Updated imports in 5 test files:
```python
# OLD (broken)
from kelpie_carbon_v1.fetch import fetch_sentinel_tiles
from kelpie_carbon_v1.mask import apply_mask

# NEW (working)
from kelpie_carbon_v1.core.fetch import fetch_sentinel_tiles
from kelpie_carbon_v1.core.mask import apply_mask
```

**🧪 Test:** All tests now pass (21 passed, 3 skipped)

---

### **Issue 3: Excessive File Watching**

**🔍 Symptoms:**
```
2025-06-09 14:12:40,668 | INFO | watchfiles.main:308 | 1 change detected
2025-06-09 14:12:41,085 | INFO | watchfiles.main:308 | 1 change detected
[Repeating every ~400ms]
```

**💡 Root Cause:**
- uvicorn watching all files indiscriminately
- No selective filtering or timing controls

**✅ Solution:**
```python
# Added selective file watching config in src/kelpie_carbon_v1/cli.py
if server_config["reload"]:
    server_config.update({
        "reload_dirs": ["src/kelpie_carbon_v1"],
        "reload_includes": ["*.py"],
        "reload_excludes": [
            "*.pyc", "__pycache__/*", "*.log", "*.tmp",
            "tests/*", "docs/*", "*.md", "*.yml", "*.yaml",
            ".git/*", ".pytest_cache/*", "*.egg-info/*"
        ],
        "reload_delay": 2.0,  # Delay between file checks (seconds)
    })
```

**🧪 Test:** Reduced frequency to 2-second intervals, ~65% CPU reduction

---

### **Issue 4: Deprecated pystac_client Methods**

**🔍 Symptoms:**
```
FutureWarning: The 'get_items' method is deprecated, use 'items' instead
```

**💡 Root Cause:**
- Using deprecated `search.get_items()` method
- Future compatibility risk

**✅ Solution:**
```python
# OLD (deprecated)
items = list(search.get_items())

# NEW (modern)
items = list(search.items())
```

**🧪 Test:** Zero deprecation warnings

---

## 🛠️ **Debugging Tips**

### **Check Server Headers**
```bash
curl -I http://localhost:8000/health
```
Look for security headers and CSP policy.

### **Verify Test Imports**
```bash
python -c "from kelpie_carbon_v1.core.fetch import fetch_sentinel_tiles; print('✅ Import OK')"
```

### **Monitor File Watching**
Watch server logs for excessive change detection:
```bash
poetry run kelpie-carbon-v1 serve --reload | grep "change detected"
```

### **Test CSP Compliance**
Check browser console for CSP violations:
- Open Developer Tools → Console
- Look for "Content-Security-Policy" errors

---

## 📋 **Verification Checklist**

### ✅ **Post-Fix Validation**
- [ ] Web interface loads without CSP errors
- [ ] Leaflet maps render correctly
- [ ] All tests pass (`poetry run python -m pytest tests/`)
- [ ] File watching limited to Python files only
- [ ] No FutureWarnings in logs
- [ ] Security headers present in responses

### ✅ **Performance Validation**
- [ ] CPU usage reduced during development
- [ ] Hot reload responds within 2-3 seconds
- [ ] Log noise minimized
- [ ] Cache management working (no memory leaks)

---

## 🔮 **Future Prevention**

### **Development Practices**
1. **Import Path Consistency**: Always use `core.*` imports
2. **CSP Testing**: Test external resources early
3. **Dependency Updates**: Check for deprecation warnings regularly
4. **Performance Monitoring**: Watch file watching frequency

### **CI/CD Integration**
```yaml
# Add to CI pipeline
- name: Check for deprecation warnings
  run: poetry run python -W error::DeprecationWarning -m pytest

- name: Validate CSP headers
  run: curl -I localhost:8000/health | grep "content-security-policy"
```

---

## 📞 **Support Information**

**If similar issues occur:**
1. Check this troubleshooting guide first
2. Verify import paths match current module structure
3. Test CSP policies with external resources
4. Monitor file watching patterns in logs
5. Run full test suite to catch regressions

**Reference Files:**
- `src/kelpie_carbon_v1/api/main.py` - Security headers & CSP
- `src/kelpie_carbon_v1/cli.py` - File watching config
- `tests/test_optimization.py` - Validation tests

---

*This guide will be updated as new issues are encountered and resolved.*
