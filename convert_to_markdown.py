"""
Convert Jupyter notebooks to Markdown files for dark-theme GitHub rendering
"""
import json
import os
from pathlib import Path

def convert_notebook_to_markdown(notebook_path, output_dir):
    """Convert a Jupyter notebook to Markdown format"""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Get notebook name
    notebook_name = Path(notebook_path).stem
    output_path = Path(output_dir) / f"{notebook_name}.md"
    
    # Create markdown content
    markdown_lines = []
    
    for cell in notebook['cells']:
        cell_type = cell['cell_type']
        source = cell.get('source', [])
        
        if cell_type == 'markdown':
            # Add markdown cell content directly
            if isinstance(source, list):
                markdown_lines.extend(source)
            else:
                markdown_lines.append(source)
            markdown_lines.append('\n')
        
        elif cell_type == 'code':
            # Add code cell with syntax highlighting
            markdown_lines.append('```python\n')
            if isinstance(source, list):
                markdown_lines.extend(source)
            else:
                markdown_lines.append(source)
            markdown_lines.append('\n```\n\n')
            
            # Add output if present
            outputs = cell.get('outputs', [])
            if outputs:
                markdown_lines.append('**Output:**\n\n')
                for output in outputs:
                    if 'text' in output:
                        markdown_lines.append('```\n')
                        text = output['text']
                        if isinstance(text, list):
                            markdown_lines.extend(text)
                        else:
                            markdown_lines.append(text)
                        markdown_lines.append('\n```\n\n')
                    elif 'data' in output:
                        # Handle different output types
                        data = output['data']
                        if 'text/plain' in data:
                            markdown_lines.append('```\n')
                            text = data['text/plain']
                            if isinstance(text, list):
                                markdown_lines.extend(text)
                            else:
                                markdown_lines.append(text)
                            markdown_lines.append('\n```\n\n')
                        if 'image/png' in data:
                            markdown_lines.append(f'![output](data:image/png;base64,{data["image/png"]})\n\n')
    
    # Write to markdown file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(markdown_lines))
    
    print(f"✅ Converted: {notebook_name}.ipynb → {notebook_name}.md")
    return output_path

# Main conversion
if __name__ == '__main__':
    # Create output directory
    output_dir = Path('chapters-dark-theme')
    output_dir.mkdir(exist_ok=True)
    
    # Convert all chapter notebooks
    notebooks = [
        'chapters/01_NumPy_Foundations.ipynb',
        'chapters/02_Pandas_DataManipulation.ipynb',
        'chapters/03_Matplotlib_Visualization.ipynb',
        'chapters/04_ScikitLearn_MachineLearning.ipynb',
    ]
    
    print("=" * 70)
    print("CONVERTING NOTEBOOKS TO MARKDOWN (Dark Theme Friendly)")
    print("=" * 70)
    
    for notebook in notebooks:
        if os.path.exists(notebook):
            convert_notebook_to_markdown(notebook, output_dir)
        else:
            print(f"❌ Not found: {notebook}")
    
    print("\n" + "=" * 70)
    print("✅ ALL CONVERSIONS COMPLETE!")
    print(f"📁 Output folder: {output_dir.absolute()}")
    print("=" * 70)
