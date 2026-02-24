import sys
import subprocess
import os

def install_package(package_name):
    """
    Installs a package using pip.
    Returns True if successful, False otherwise.
    """
    try:
        # We use sys.executable to ensure we use the same python interpreter.
        # We run pip as a module (-m pip) to avoid path issues.
        command = [sys.executable, "-m", "pip", "install", package_name]
        print(f"Running: {' '.join(command)}")
        subprocess.check_call(command)
        return True
    except subprocess.CalledProcessError:
        return False

def get_clean_package_name(requirement_line):
    """
    Extracts the package name from a requirement line, removing version specifiers and URLs.
    Example: 'pandas==2.2.2' -> 'pandas'
             'package @ https://...' -> 'package'
    """
    # List of operators that start a version specifier or URL
    operators = ['==', '>=', '<=', '~=', '!=', '>', '<', '@']
    
    clean_name = requirement_line.strip()
    # Remove environment markers first (e.g., "; python_version < '3.0'")
    clean_name = clean_name.split(';')[0].strip()
    
    for op in operators:
        if op in clean_name:
            clean_name = clean_name.split(op)[0].strip()
            
    return clean_name

def main():
    # Determine the directory where this script is located to find requirements.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file = os.path.join(script_dir, 'requirements.txt')
    
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found.")
        sys.exit(1)

    print(f"Reading requirements from: {requirements_file}")
    with open(requirements_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        print(f"\n" + "="*60)
        print(f"Processing: {line}")
        
        # Attempt to install the exact requirement line
        success = install_package(line)
        
        if not success:
            print(f"\n[!] Failed to install: {line}")
            
            # Suggest alternative (clean package name without version constraints)
            alternative = get_clean_package_name(line)
            
            if alternative and alternative != line:
                print(f"[?] Alternative suggestion: Install '{alternative}'")
                print("    (This removes version constraints to find a compatible version)")
                
                response = input(f"    Do you want to try installing '{alternative}'? (y/n): ").strip().lower()
                
                if response == 'y':
                    print(f"    Attempting to install alternative: {alternative}")
                    if install_package(alternative):
                        print(f"[+] Successfully installed alternative: {alternative}")
                    else:
                        print(f"[-] Failed to install alternative: {alternative}")
                else:
                    print("    Skipping package.")
            else:
                print("    No simple alternative found. Skipping package.")

if __name__ == "__main__":
    main()