#!/usr/bin/env python3
"""
Test Suite Consolidation Analysis Script

This script analyzes the existing test suite to:
1. Identify duplicate/near-duplicate tests
2. Group tests by function-under-test 
3. Generate parameterized test consolidations
4. Create a consolidation plan
"""

import os
import ast
import re
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import hashlib


class TestAnalyzer:
    def __init__(self, test_root: str = "tests"):
        self.test_root = Path(test_root)
        self.test_files = []
        self.test_functions = []
        self.duplicate_groups = defaultdict(list)
        self.consolidation_plan = {}
        
    def scan_test_files(self):
        """Scan all test files in the test directory"""
        print(f"Scanning test files in {self.test_root}...")
        
        for root, dirs, files in os.walk(self.test_root):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    file_path = Path(root) / file
                    self.test_files.append(file_path)
                    
        print(f"Found {len(self.test_files)} test files")
        return self.test_files
    
    def parse_test_file(self, file_path: Path) -> List[Dict]:
        """Parse a test file to extract test functions and their characteristics"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            tests = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_info = self.analyze_test_function(node, file_path, content)
                    if test_info:
                        tests.append(test_info)
                        
            return tests
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
    
    def analyze_test_function(self, node: ast.FunctionDef, file_path: Path, content: str) -> Dict:
        """Analyze a test function to extract key characteristics"""
        # Get the source code lines for this function
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
        
        function_source = '\n'.join(lines[start_line:end_line])
        
        # Extract test characteristics
        test_info = {
            'name': node.name,
            'file': str(file_path),
            'source': function_source,
            'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
            'assertions': self.extract_assertions(node),
            'function_calls': self.extract_function_calls(node),
            'imports_used': self.extract_imports_from_function(function_source),
            'signature_hash': self.compute_signature_hash(node, function_source),
            'is_parameterized': any('parametrize' in str(d) for d in node.decorator_list)
        }
        
        return test_info
    
    def extract_assertions(self, node: ast.FunctionDef) -> List[str]:
        """Extract assertion patterns from a test function"""
        assertions = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                # Convert assertion back to source-like string
                assertion_str = ast.dump(child.test)
                assertions.append(assertion_str)
                
        return assertions
    
    def extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract function calls being tested"""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
                    
        return calls
    
    def extract_imports_from_function(self, source: str) -> List[str]:
        """Extract imports or modules used in function source"""
        imports = []
        
        # Look for common patterns
        import_patterns = [
            r'from\s+(\w+(?:\.\w+)*)\s+import',
            r'import\s+(\w+(?:\.\w+)*)',
            r'(\w+)\.\w+\(',  # module.function() calls
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, source)
            imports.extend(matches)
            
        return list(set(imports))
    
    def compute_signature_hash(self, node: ast.FunctionDef, source: str) -> str:
        """Compute a hash representing the test's structure/signature"""
        # Normalize the source code
        normalized = re.sub(r'\s+', ' ', source)  # Normalize whitespace
        normalized = re.sub(r'#.*', '', normalized)  # Remove comments
        normalized = re.sub(r'""".*?"""', '', normalized, flags=re.DOTALL)  # Remove docstrings
        
        # Create signature from key elements
        signature_elements = [
            node.name.replace('test_', ''),  # Remove test prefix
            str(sorted(self.extract_function_calls(node))),
            str(sorted(self.extract_assertions(node))),
        ]
        
        signature = '|'.join(signature_elements)
        return hashlib.md5(signature.encode()).hexdigest()[:12]
    
    def analyze_all_tests(self):
        """Analyze all test files and identify patterns"""
        print("Analyzing test functions...")
        
        for file_path in self.test_files:
            tests = self.parse_test_file(file_path)
            self.test_functions.extend(tests)
            
        print(f"Found {len(self.test_functions)} test functions")
        
        # Group by various criteria
        self.group_by_function_under_test()
        self.group_by_similarity()
        self.identify_parameterization_candidates()
        
    def group_by_function_under_test(self):
        """Group tests by the main function they're testing"""
        function_groups = defaultdict(list)
        
        for test in self.test_functions:
            # Extract the main function being tested from test name
            test_name = test['name']
            
            # Remove test_ prefix and common suffixes
            function_name = test_name.replace('test_', '')
            function_name = re.sub(r'_\d+$', '', function_name)  # Remove numeric suffixes
            function_name = re.sub(r'_(with|using|when|should|can|if)_.*', '', function_name)
            
            function_groups[function_name].append(test)
            
        # Filter groups with multiple tests (potential duplicates)
        self.duplicate_groups['by_function'] = {
            func: tests for func, tests in function_groups.items() 
            if len(tests) > 1
        }
        
        print(f"Found {len(self.duplicate_groups['by_function'])} function groups with potential duplicates")
    
    def group_by_similarity(self):
        """Group tests by structural similarity"""
        signature_groups = defaultdict(list)
        
        for test in self.test_functions:
            signature_groups[test['signature_hash']].append(test)
            
        # Filter groups with multiple tests
        self.duplicate_groups['by_signature'] = {
            sig: tests for sig, tests in signature_groups.items() 
            if len(tests) > 1
        }
        
        print(f"Found {len(self.duplicate_groups['by_signature'])} signature groups with potential duplicates")
    
    def identify_parameterization_candidates(self):
        """Identify tests that can be parameterized"""
        candidates = {}
        
        for group_name, tests in self.duplicate_groups['by_function'].items():
            if len(tests) >= 3:  # Need at least 3 tests to justify parameterization
                # Analyze if tests differ only in input values
                param_potential = self.analyze_parameterization_potential(tests)
                if param_potential['can_parameterize']:
                    candidates[group_name] = param_potential
                    
        self.duplicate_groups['parameterization_candidates'] = candidates
        print(f"Found {len(candidates)} parameterization candidates")
    
    def analyze_parameterization_potential(self, tests: List[Dict]) -> Dict:
        """Analyze if a group of tests can be parameterized"""
        # Look for common patterns in test names and source
        test_names = [test['name'] for test in tests]
        
        # Check if tests follow similar patterns
        common_assertions = set(tests[0]['assertions'])
        common_calls = set(tests[0]['function_calls'])
        
        for test in tests[1:]:
            common_assertions &= set(test['assertions'])
            common_calls &= set(test['function_calls'])
            
        # Estimate parameterization potential
        can_parameterize = (
            len(common_assertions) > 0 and 
            len(common_calls) > 0 and
            len(tests) >= 3
        )
        
        return {
            'can_parameterize': can_parameterize,
            'test_count': len(tests),
            'common_assertions': len(common_assertions),
            'common_calls': len(common_calls),
            'tests': tests,
            'estimated_reduction': len(tests) - 1  # All tests become 1 parameterized test
        }
    
    def generate_consolidation_plan(self):
        """Generate a plan for test consolidation"""
        plan = {
            'baseline': {
                'total_test_files': len(self.test_files),
                'total_test_functions': len(self.test_functions),
                'total_lines': sum(len(test['source'].split('\n')) for test in self.test_functions)
            },
            'analysis': {
                'function_groups': len(self.duplicate_groups['by_function']),
                'signature_groups': len(self.duplicate_groups['by_signature']),
                'parameterization_candidates': len(self.duplicate_groups['parameterization_candidates'])
            },
            'consolidation_opportunities': [],
            'estimated_savings': {
                'tests_removed': 0,
                'files_removed': 0,
                'reduction_percentage': 0
            }
        }
        
        # Calculate consolidation opportunities
        total_removable = 0
        
        for group_name, candidate in self.duplicate_groups['parameterization_candidates'].items():
            opportunity = {
                'type': 'parameterization',
                'function': group_name,
                'current_tests': candidate['test_count'],
                'consolidated_tests': 1,
                'reduction': candidate['estimated_reduction'],
                'files_affected': list(set(test['file'] for test in candidate['tests']))
            }
            plan['consolidation_opportunities'].append(opportunity)
            total_removable += candidate['estimated_reduction']
            
        # Find simple duplicates (similar signatures but not parameterizable)
        for sig, tests in self.duplicate_groups['by_signature'].items():
            if len(tests) > 1:
                # Check if this group is NOT already in parameterization candidates
                group_functions = set()
                for test in tests:
                    func_name = test['name'].replace('test_', '')
                    func_name = re.sub(r'_\d+$', '', func_name)
                    group_functions.add(func_name)
                
                if not any(func in self.duplicate_groups['parameterization_candidates'] 
                          for func in group_functions):
                    opportunity = {
                        'type': 'duplicate_removal',
                        'signature': sig,
                        'current_tests': len(tests),
                        'consolidated_tests': 1,
                        'reduction': len(tests) - 1,
                        'files_affected': list(set(test['file'] for test in tests))
                    }
                    plan['consolidation_opportunities'].append(opportunity)
                    total_removable += len(tests) - 1
        
        # Calculate estimated savings
        plan['estimated_savings']['tests_removed'] = total_removable
        plan['estimated_savings']['reduction_percentage'] = (
            total_removable / plan['baseline']['total_test_functions'] * 100
        )
        
        self.consolidation_plan = plan
        return plan
    
    def save_analysis(self, output_file: str = "test_analysis.json"):
        """Save the analysis results to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.consolidation_plan, f, indent=2)
        print(f"Analysis saved to {output_file}")
    
    def print_summary(self):
        """Print a summary of the analysis"""
        plan = self.consolidation_plan
        
        print("\n" + "="*80)
        print("TEST CONSOLIDATION ANALYSIS SUMMARY")
        print("="*80)
        print(f"📊 Baseline:")
        print(f"   • Test files: {plan['baseline']['total_test_files']}")
        print(f"   • Test functions: {plan['baseline']['total_test_functions']}")
        print(f"   • Total lines: {plan['baseline']['total_lines']}")
        
        print(f"\n🔍 Analysis Results:")
        print(f"   • Function groups: {plan['analysis']['function_groups']}")
        print(f"   • Signature groups: {plan['analysis']['signature_groups']}")  
        print(f"   • Parameterization candidates: {plan['analysis']['parameterization_candidates']}")
        
        print(f"\n⚡ Consolidation Opportunities:")
        for opp in plan['consolidation_opportunities'][:10]:  # Show top 10
            print(f"   • {opp['type']}: {opp.get('function', opp.get('signature', 'N/A')[:12])} "
                  f"({opp['current_tests']} → {opp['consolidated_tests']} tests, "
                  f"-{opp['reduction']} tests)")
        
        if len(plan['consolidation_opportunities']) > 10:
            print(f"   ... and {len(plan['consolidation_opportunities']) - 10} more opportunities")
            
        print(f"\n💾 Estimated Savings:")
        print(f"   • Tests removed: {plan['estimated_savings']['tests_removed']}")
        print(f"   • Reduction: {plan['estimated_savings']['reduction_percentage']:.1f}%")
        
        target_reduction = 50  # 50% target
        if plan['estimated_savings']['reduction_percentage'] >= target_reduction:
            print(f"   ✅ Target {target_reduction}% reduction ACHIEVABLE!")
        else:
            print(f"   ⚠️  Target {target_reduction}% reduction may require additional consolidation")


def main():
    """Main analysis function"""
    print("Starting test suite consolidation analysis...")
    
    analyzer = TestAnalyzer()
    analyzer.scan_test_files()
    analyzer.analyze_all_tests()
    analyzer.generate_consolidation_plan()
    analyzer.save_analysis()
    analyzer.print_summary()
    
    print("\n✅ Analysis complete! Check test_analysis.json for detailed results.")
    print("Next steps:")
    print("1. Review consolidation opportunities")
    print("2. Implement parameterized tests")
    print("3. Remove duplicate tests")
    print("4. Run coverage verification")


if __name__ == "__main__":
    main() 
