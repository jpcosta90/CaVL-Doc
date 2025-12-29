
import csv
import os

input_file = '/home/joaopaulo/Projects/CaVL-Doc/docs/paper/search/scopus_export_Dec 27-2025_f593f4df-f6d7-4146-bcc0-2c6ff05fc25c.csv'
output_file = '/home/joaopaulo/Projects/CaVL-Doc/docs/paper/search/relevant_articles_filtered.csv'

# Keywords to score relevance
# We want papers that discuss:
# 1. Zero-shot/Few-shot gaps in documents
# 2. Layout vs Semantic importance
# 3. Efficiency/Cost of LVLMs vs Specialized models
# 4. Embeddings / Metric Learning / Retrieval as a solution
# 5. Specific high-value targets: "DocVLM", "ColPali", "Embeddings"

high_value_terms = ["DocVLM", "ColPali", "Visual Document Matching", "Metric Learning", "Embeddings"]
relevance_terms = [
    "zero-shot", "few-shot", "layout", "structural", "structure", 
    "efficiency", "cost", "latency", "generative", "discriminative",
    "retrieval", "contrastive", "visual document understanding", "vrdu",
    "fine-grained", "unseen classes"
]

def score_article(title, abstract):
    text = (title + " " + abstract).lower()
    score = 0
    
    # 1. Check for high value specific papers
    for term in high_value_terms:
        if term.lower() in text:
            score += 10 # Massive boost
            
    # 2. Check for general relevance terms
    found_terms = 0
    for term in relevance_terms:
        if term in text:
            score += 1
            found_terms += 1
            
    # 3. Specific narrative check: "Layout" AND "Zero-shot/Few-shot" (The Gap)
    if "layout" in text and ("zero-shot" in text or "few-shot" in text):
        score += 3
        
    # 4. Specific narrative check: "Generative" AND ("Efficiency" or "Cost") (The Paradox)
    if "generative" in text and ("efficien" in text or "cost" in text or "latency" in text):
        score += 3

    return score

selected_articles = []

with open(input_file, 'r', encoding='utf-8') as f_in:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames
    
    for row in reader:
        title = row.get('Title', '')
        abstract = row.get('Abstract', '')
        
        # Calculate score
        score = score_article(title, abstract)
        
        # Threshold: At least one strong signal or multiple weak ones
        # If it mentions DocVLM or ColPali, it gets 10+, so it's in.
        # If it's a generic paper, it needs a combination of terms.
        if score >= 4: 
            selected_articles.append((score, row))

# Sort by score descending
selected_articles.sort(key=lambda x: x[0], reverse=True)


# Write output with selected columns only
output_fieldnames = ['Title', 'Year', 'DOI', 'Link']

with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.DictWriter(f_out, fieldnames=output_fieldnames)
    writer.writeheader()
    for score, row in selected_articles:
        # Create a clean row dictionary with only the desired fields
        clean_row = {
            'Title': row.get('Title', ''),
            'Year': row.get('Year', ''),
            'DOI': row.get('DOI', ''),
            'Link': row.get('Link', '')
        }
        writer.writerow(clean_row)

print(f"Filtered {len(selected_articles)} relevant articles from source.")
for s, r in selected_articles[:5]:
    print(f"[{s}] {r['Title']}")
