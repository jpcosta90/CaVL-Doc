import zipfile
import re
import xml.etree.ElementTree as ET

xlsx_path = "docs/paper/references/introduction-and-state-of-art/relevant_articles_filtered.xlsx"

def read_xlsx_strings(path):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            # Try to read shared strings
            if 'xl/sharedStrings.xml' in z.namelist():
                xml_content = z.read('xl/sharedStrings.xml')
                root = ET.fromstring(xml_content)
                # The namespace usually is something like {http://schemas.openxmlformats.org/spreadsheetml/2006/main}
                # We can just iterate all 't' elements which contain text
                strings = []
                for elem in root.iter():
                    if elem.tag.endswith('t'):
                        if elem.text:
                            strings.append(elem.text)
                
                print(f"--- Extracted {len(strings)} strings from sharedStrings.xml ---")
                for i, s in enumerate(strings):
                    print(f"[{i}] {s}")
            else:
                print("No sharedStrings.xml found. Optimised xlsx?")
                
            # If it's inline strings (rare but possible), they would be in sheet*.xml
            # But let's start with sharedStrings.
    except Exception as e:
        print(f"Error reading xlsx: {e}")

if __name__ == "__main__":
    read_xlsx_strings(xlsx_path)
