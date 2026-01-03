
import requests
import pandas as pd
import time
import os

# Define search queries based on our 3 categories
QUERIES = {
    "1_BigField": {
        "query": '("Document Understanding"|"Document Image"|"Visually Rich Documents")',
        "year": "2021-2025",
        "limit": 100
    },
    "2_ModernWave": {
        "query": '("Vision-Language"|"Multimodal"|"Large Language Model") "Document"',
        "year": "2023-2025",
        "limit": 100
    },
    "3_MethodNiche": {
        "query": '("Zero-shot"|"Few-shot"|"Embedding"|"Distillation") "Document Classification"',
        "year": "2020-2025",
        "limit": 100
    }
}

OUTPUT_DIR = "docs/paper/search"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_papers(key, config):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": config["query"],
        "year": config["year"],
        "fields": "title,abstract,citationCount,year,authors",
        "limit": config["limit"]
    }
    
    print(f"Fetching '{key}' (Year: {config['year']})...")
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return None
            
        data = response.json()
        if "data" not in data:
            print("No data returned.")
            return None
            
        papers = data["data"]
        print(f"  -> Found {len(papers)} papers.")
        
        # Convert to DataFrame matching Scopus structure for compatibility
        # Scopus cols: "Title", "Cited by", "Abstract", "Year"
        rows = []
        for p in papers:
            rows.append({
                "Title": p.get("title", ""),
                "Cited by": p.get("citationCount", 0),
                "Abstract": p.get("abstract", ""),
                "Year": p.get("year", ""),
                # Semantic scholar doesn't have keywords easily in search, but Abstract helps
                "Author keywords": "" 
            })
            
        return pd.DataFrame(rows)
        
    except Exception as e:
        print(f"Exception: {e}")
        return None

def main():
    for key, config in QUERIES.items():
        df = fetch_papers(key, config)
        if df is not None:
            filename = os.path.join(OUTPUT_DIR, f"semantic_scholar_{key}.csv")
            df.to_csv(filename, index=False)
            print(f"  -> Saved to {filename}")
        
        # Be nice to the API
        time.sleep(2)

if __name__ == "__main__":
    main()
