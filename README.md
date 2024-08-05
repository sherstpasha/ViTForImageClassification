# update_readme.py
example_file = 'example.py'
readme_file = 'README.md'

with open(example_file, 'r') as ef, open(readme_file, 'w') as rf:
    rf.write("# Project Title\n\n")
    rf.write("## Example Usage\n\n")
    rf.write("This example demonstrates how to use the code.\n\n")
    rf.write("```python\n")
    rf.write(ef.read())
    rf.write("\n```\n")
