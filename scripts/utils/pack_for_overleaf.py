import zipfile
import os
import sys

def pack_for_overleaf(output_filename="cavl_doc_overleaf.zip"):
    source_dir = "docs"
    
    # Patterns/Files to exclude
    # 1. Compilation artifacts
    exclude_extensions = {'.aux', '.log', '.out', '.toc', '.synctex.gz', '.fls', '.fdb_latexmk', '.bbl', '.blg', '.dvi'}
    # 2. System files
    exclude_files = {'.DS_Store', 'Thumbs.db'}
    # 3. Heavy/Unneeded folders
    # We DO NOT need the PDF references in Overleaf, only the .bib file.
    exclude_folders = {'__pycache__', '.ipynb_checkpoints', 'references', '.git'} 

    print(f"Packing '{source_dir}' into '{output_filename}'...")
    
    file_count = 0
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Modify dirs in-place to skip excluded folders
            dirs[:] = [d for d in dirs if d not in exclude_folders]
            
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in exclude_extensions or file in exclude_files:
                    continue
                
                # Construct full path
                file_path = os.path.join(root, file)
                
                # Construct arcname (relative path inside zip)
                # We strip "docs/" so "paper" and "assets" are at the root of the zip
                # This ensures consistent relative paths for Overleaf
                arcname = os.path.relpath(file_path, start=source_dir)
                
                print(f"  Adding: {arcname}")
                zipf.write(file_path, arcname)
                file_count += 1
                
    print(f"\nSuccess! Created {output_filename} with {file_count} files.")
    print("Upload this file directly to Overleaf via 'New Project' -> 'Upload Project'.")

if __name__ == "__main__":
    pack_for_overleaf()
