from pathlib import Path


def getProjectRoot():
    # Current file location
    current_file = Path(__file__).resolve()
    # Loop to ascend to the project root identified by a marker, e.g., 'requirements.txt'
    project_root = current_file
    while not (project_root / 'requirements.txt').exists():
        if project_root.parent == project_root:
            raise RuntimeError('requirements.txt directory not found, cannot determine project root')
        project_root = project_root.parent
    return project_root
