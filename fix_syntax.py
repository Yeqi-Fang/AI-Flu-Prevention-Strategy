"""
Script to fix syntax errors in modelDQN.py
"""
import re

# Read the current file
with open('src/rl/modelDQN.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the syntax errors by adding proper line breaks
content = re.sub(r'(\w+)\s*([A-Za-z_][\w.]*\s*=)', r'\1\n        \2', content)
content = re.sub(r'return\s+([a-zA-Z])', r'return\n\n        \1', content)

# Write the fixed content back
with open('src/rl/modelDQN.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed syntax errors in modelDQN.py")
