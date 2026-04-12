import os
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
README_PATH = REPO_ROOT / "README.md"

IGNORE_DIRS = {
    ".git", "__pycache__", "logs", "figs", "checkpoints", 
    ".venv", "venv", ".sisyphus", ".ruff_cache", ".mypy_cache"
}

START_MARKER = "<!-- STRUCTURE_START -->"
END_MARKER = "<!-- STRUCTURE_END -->"

# ── Logic ───────────────────────────────────────────────────────────────────

def generate_tree(dir_path: Path, prefix: str = "") -> list[str]:
    """Recursively build ASCII tree lines."""
    lines = []
    
    # Get sorted entries, directories first
    contents = sorted(
        [d for d in dir_path.iterdir() if d.name not in IGNORE_DIRS and not d.name.startswith(".")],
        key=lambda x: (not x.is_dir(), x.name)
    )
    
    for i, path in enumerate(contents):
        is_last = (i == len(contents) - 1)
        connector = "└── " if is_last else "├── "
        
        lines.append(f"{prefix}{connector}{path.name}{'/' if path.is_dir() else ''}")
        
        if path.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(generate_tree(path, prefix + extension))
            
    return lines

def update_readme():
    if not README_PATH.exists():
        print(f"Error: {README_PATH} not found.")
        return

    # Generate the new tree
    tree_lines = ["```text", "autoresearch-mlx/"] + generate_tree(REPO_ROOT) + ["```"]
    new_structure = "\n".join(tree_lines)

    # Read current README
    with open(README_PATH, "r") as f:
        content = f.read()

    # Find markers
    start_idx = content.find(START_MARKER)
    end_idx = content.find(END_MARKER)

    if start_idx == -1 or end_idx == -1:
        print(f"Warning: Markers {START_MARKER} or {END_MARKER} not found in README.md.")
        print("Appending to end of file instead.")
        with open(README_PATH, "a") as f:
            f.write(f"\n\n{START_MARKER}\n{new_structure}\n{END_MARKER}\n")
        return

    # Replace content between markers
    new_content = (
        content[:start_idx + len(START_MARKER)] +
        "\n" + new_structure + "\n" +
        content[end_idx:]
    )

    with open(README_PATH, "w") as f:
        f.write(new_content)
    
    print("README.md structure section updated successfully.")

if __name__ == "__main__":
    update_readme()
