# 🚀 CI Configuration Guide

## 📋 Current CI Setup

The CI is configured to run in the following scenarios:

### **🔄 Triggers**

1. **Push to `master` branch** - Runs CI on every merge into master
2. **Pull Requests** - Runs CI on every PR, **EXCEPT** when targeting the `master` branch

### **⚙️ Configuration Details**

```yaml
on:
  push:
    branches: [master]           # Only run on pushes to master
  pull_request:
    branches-ignore: [master]   # Run on PRs, but ignore PRs targeting master
```

## 🎯 What This Means

### **✅ CI Will Run:**
- ✅ When you push directly to `master` branch
- ✅ When you create a PR from `feature-branch` → `develop`
- ✅ When you create a PR from `feature-branch` → `staging`
- ✅ When you create a PR from `develop` → `staging`
- ✅ When you merge any PR into `master`

### **❌ CI Will NOT Run:**
- ❌ When you create a PR from `feature-branch` → `master`
- ❌ When you push to non-master branches (like `develop`, `feature-branch`)
- ❌ When you create PRs targeting `master` directly

## 🔧 CI Pipeline Steps

### **1. Environment Setup**
- Uses Ubuntu latest
- Tests on Python 3.8, 3.9, 3.10
- Uses latest GitHub Actions (v4)

### **2. Dependencies Installation**
- Upgrades pip
- Installs `uv` package manager
- Installs project dependencies

### **3. Code Quality Checks**
- Runs pre-commit hooks
- Formatting and linting (ruff)
- Security checks (bandit)
- Typo checking (codespell)

### **4. Testing**
- Installs test dependencies
- Runs football prediction tests
- Tests model functionality
- Tests app components
- Tests data creation

## 🚀 Workflow Examples

### **Scenario 1: Feature Development**
```bash
# 1. Create feature branch
git checkout -b feature/new-prediction-algorithm

# 2. Make changes and commit
git add .
git commit -m "Add new prediction algorithm"

# 3. Push to remote
git push origin feature/new-prediction-algorithm

# 4. Create PR to develop
# → CI will run automatically ✅
```

### **Scenario 2: Merge to Master**
```bash
# 1. Merge PR into master
# → CI will run automatically ✅

# 2. Or push directly to master
git push origin master
# → CI will run automatically ✅
```

### **Scenario 3: Direct Master PR (Blocked)**
```bash
# 1. Create PR from feature → master
# → CI will NOT run ❌
# This prevents direct changes to master
```

## 🛡️ Benefits of This Configuration

### **1. Master Branch Protection**
- Prevents direct PRs to master
- Ensures all changes go through proper review process
- Master branch stays stable

### **2. Development Workflow**
- CI runs on all development branches
- Catches issues early in development
- Ensures code quality before merging

### **3. Release Process**
- CI runs on every merge to master
- Ensures production code is tested
- Validates all changes before deployment

## 🔄 Alternative Configurations

### **Option 1: Run CI on All Branches**
```yaml
on: [push, pull_request]
```

### **Option 2: Run CI Only on PRs**
```yaml
on:
  pull_request:
```

### **Option 3: Run CI on Specific Branches**
```yaml
on:
  push:
    branches: [master, develop]
  pull_request:
    branches: [master, develop]
```

## 📊 CI Status Monitoring

### **Check CI Status:**
1. Go to your GitHub repository
2. Click on "Actions" tab
3. View CI runs and their status
4. Check logs for any failures

### **Common CI Issues:**
- **Linting errors** - Fix code formatting
- **Test failures** - Fix broken tests
- **Dependency issues** - Update requirements
- **Security warnings** - Address security issues

## 🎉 Success Criteria

Your CI is working correctly when:
- ✅ All tests pass
- ✅ No linting errors
- ✅ No security warnings
- ✅ Code formatting is correct
- ✅ All dependencies install successfully

## 🚀 Next Steps

1. **Test the CI** - Create a test PR to verify it works
2. **Monitor CI runs** - Check the Actions tab regularly
3. **Fix any issues** - Address CI failures promptly
4. **Optimize if needed** - Adjust configuration as needed

The CI is now configured to protect your master branch while ensuring all development work is properly tested! 🎯
