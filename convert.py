import nbformat
from nbconvert import PythonExporter

# 读取 notebook 文件
with open('example.ipynb') as f:
    notebook = nbformat.read(f, as_version=4)

# 使用 PythonExporter 转换为 Python 脚本
exporter = PythonExporter()
script, _ = exporter.from_notebook_node(notebook)

# 保存为 .py 文件
with open('example.py', 'w') as f:
    f.write(script)
