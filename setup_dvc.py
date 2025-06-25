import os
import subprocess

def setup_dvc():
    # Initialize DVC if not already initialized
    if not os.path.exists('.dvc'):
        subprocess.run(['dvc', 'init'])
        print("DVC initialized")
    else:
        print("DVC already initialized")

    # Ensure DVC entries are in .gitignore
    gitignore_entries = [
        '/Data/Loan.csv',  # The actual data file
        '/.dvc/cache',     # DVC cache directory
        '/dvc.lock',       # DVC lock file
        '*.tmp',           # DVC temporary files
    ]
    
    # Create or update .gitignore
    gitignore_path = '.gitignore'
    existing_entries = []
    
    # Read existing entries if file exists
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            existing_entries = [line.strip() for line in f.readlines()]
    
    # Add new entries that don't already exist
    with open(gitignore_path, 'a') as f:
        for entry in gitignore_entries:
            if entry not in existing_entries:
                f.write(f"{entry}\n")
                print(f"Added '{entry}' to .gitignore")

    # Add data file to DVC
    data_file = 'Data/Loan.csv'
    if os.path.exists(data_file):
        subprocess.run(['dvc', 'add', data_file])
        print(f"Added {data_file} to DVC tracking")
        
        # Add the .dvc file to git
        subprocess.run(['git', 'add', f"{data_file}.dvc", '.gitignore'])
        subprocess.run(['git', 'commit', '-m', f"Add {data_file} to DVC"])
        print(f"Committed {data_file}.dvc to git")
    else:
        print(f"Error: {data_file} not found")

if __name__ == "__main__":
    setup_dvc()