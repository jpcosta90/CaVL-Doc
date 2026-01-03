
import pandas as pd
import glob
import re
import collections
from sklearn.feature_extraction.text import CountVectorizer

def analyze_titles_and_keywords(data_path="docs/paper/search/*.csv"):
    csv_files = glob.glob(data_path)
    all_data = []

    print(f"Loading files from {data_path}...")
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Normalize column names just in case
            df.columns = [c.strip() for c in df.columns]
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No data found.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter for relevant columns
    if 'Cited by' not in combined_df.columns or 'Title' not in combined_df.columns:
        print("Required columns 'Cited by' or 'Title' missing.")
        # Attempt to map from likely alternatives if needed, but Scopus export usually has these.
        return

    # Clean Citations (NaN -> 0)
    combined_df['Cited by'] = combined_df['Cited by'].fillna(0)
    # Ensure numeric
    combined_df['Cited by'] = pd.to_numeric(combined_df['Cited by'], errors='coerce').fillna(0)

    # Filter Outliers (User Request: "Not huge successes like Attention Is All You Need")
    # Let's verify standard deviation or set a hard cap based on domain knowledge.
    # For a niche field, > 500 citations/year might be an outlier for a "normal impact" paper analysis.
    # But let's look at the distribution first if we were interactive. 
    # Here, I'll set a reasonable cap to exclude mega-blockbusters if they exist in the dataset.
    # Assuming the user wants 'achievable high impact' (e.g. 20-100 citations).
    
    # Let's categorize:
    # High Impact (Realistic): Top 10-25% of the remaining distribution?
    # Or simply: Cited by > 5 AND Cited by < 500 (just an arbitrary range to filter noise and aliens)
    
    refined_df = combined_df[combined_df['Cited by'] < 2000] # Safe upper bound
    refined_df = refined_df[refined_df['Cited by'] > 0] # At least some impact

    # Define High vs Low
    high_threshold = refined_df['Cited by'].quantile(0.75)
    low_threshold = refined_df['Cited by'].quantile(0.25)
    
    high_impact_df = refined_df[refined_df['Cited by'] >= high_threshold]
    low_impact_df = refined_df[refined_df['Cited by'] <= low_threshold]

    print(f"Analyzing Contrast:")
    print(f"  - High Impact (Top 25%): > {high_threshold} citations (n={len(high_impact_df)})")
    print(f"  - Low Impact (Bottom 25%): < {low_threshold} citations (n={len(low_impact_df)})")

    def get_group_stats(df, name):
        stats = {}
        # 1. Length
        lens = df['Title'].apply(lambda x: len(str(x).split()))
        stats['avg_len'] = lens.mean()
        
        # 2. Colon Usage
        has_colon = df['Title'].apply(lambda x: ':' in str(x)).sum()
        stats['colon_pct'] = (has_colon / len(df)) * 100
        
        # 3. Start Words
        start_words = df['Title'].apply(lambda x: str(x).split()[0].rstrip(':').lower())
        stats['top_starts'] = collections.Counter(start_words).most_common(5)
        
        # 4. Connectors
        target_connectors = ['for', 'via', 'using', 'with', 'based', 'towards']
        conn_stats = {}
        for conn in target_connectors:
            count = df['Title'].apply(lambda x: f" {conn} " in str(x).lower()).sum()
            conn_stats[conn] = (count / len(df)) * 100
        stats['connectors'] = conn_stats
        
        return stats

    high_stats = get_group_stats(high_impact_df, "High")
    low_stats = get_group_stats(low_impact_df, "Low")
    
    print("\n" + "="*40)
    print("      CONTRAST ANALYSIS (High vs Low)")
    print("="*40)
    
    print(f"\n1. Title Length:")
    print(f"   - High Impact: {high_stats['avg_len']:.1f} words")
    print(f"   - Low Impact:  {low_stats['avg_len']:.1f} words")
    
    print(f"\n2. Structure (Colon Usage):")
    print(f"   - High Impact: {high_stats['colon_pct']:.1f}%")
    print(f"   - Low Impact:  {low_stats['colon_pct']:.1f}%")
    
    print(f"\n3. Connector Usage (The 'Why' vs 'How'):")
    for conn in ['for', 'with', 'using', 'via']:
        h = high_stats['connectors'][conn]
        l = low_stats['connectors'][conn]
        diff = h - l
        marker = "🔥" if diff > 5 else "🔻" if diff < -5 else " "
        print(f"   - {conn.ljust(6)}: High={h:.1f}% | Low={l:.1f}% | Diff={diff:+.1f}% {marker}")

    print(f"\n4. Top Starting Words:")
    print(f"   [High Impact]            [Low Impact]")
    for i in range(5):
        h_w, h_c = high_stats['top_starts'][i] if i < len(high_stats['top_starts']) else ("-", 0)
        l_w, l_c = low_stats['top_starts'][i] if i < len(low_stats['top_starts']) else ("-", 0)
        print(f"   {i+1}. {h_w.ljust(15)} ({h_c}) | {l_w.ljust(15)} ({l_c})")

    # ANALYSIS 5: CATEGORY CHECK (Brand vs Verb vs Generic)
    def classify_start(title):
        title = str(title).strip()
        first_word = title.split()[0].lower()
        
        # Heuristic 1: Branding (Has Colon AND first part is short < 3 words)
        if ':' in title:
            pre_colon = title.split(':')[0]
            if len(pre_colon.split()) <= 3:
                return "Brand/Acronym"
        
        # Heuristic 2: Action Verbs
        verbs = ['enhancing', 'improving', 'boosting', 'evaluating', 'comparing', 'learning', 
                 'pre-training', 'fine-tuning', 'towards', 'optimizing', 'leveraging', 'revisiting']
        if first_word in verbs or first_word.endswith('ing'):
            return "Action Verb"
            
        return "Generic/Noun"

    print("\n" + "="*40)
    print("      START STRATEGY SHOWDOWN")
    print("="*40)
    
    refined_df['Strategy'] = refined_df['Title'].apply(classify_start)
    strategy_stats = refined_df.groupby('Strategy')['Cited by'].agg(['mean', 'count', 'median'])
    
    # Sort by Mean Citations to see what wins
    strategy_stats = strategy_stats.sort_values('mean', ascending=False)
    
    print(strategy_stats)
    print("\n -> 'Brand/Acronym' includes titles like 'ModelName: ...'")
    print(" -> 'Action Verb' includes titles starting with 'Enhancing', 'Comparing', etc.")

    print(strategy_stats)
    print("\n -> 'Brand/Acronym' includes titles like 'ModelName: ...'")
    print(" -> 'Action Verb' includes titles starting with 'Enhancing', 'Comparing', etc.")

    # ANALYSIS 6: KEYWORD CONTRAST (High vs Low)
    print("\n" + "="*40)
    print("      KEYWORD CONTRAST (Discriminating Terms)")
    print("="*40)
    
    def get_keywords(df):
        all_k = []
        # Try 'Author keywords' then 'Index Keywords' then fallback to extracting from Title N-grams if empty
        # For this dataset (Semantic Scholar), we might need to rely on Title N-grams if 'Author keywords' is empty
        # Let's check Title N-grams first as a proxy since Semantic Scholar CSV might have empty keywords
        
        # Using Title N-grams (bi-grams and tri-grams)
        vec = CountVectorizer(stop_words='english', ngram_range=(2, 3), min_df=2)
        try:
            X = vec.fit_transform(df['Title'].astype(str))
            counts = X.toarray().sum(axis=0)
            return dict(zip(vec.get_feature_names_out(), counts))
        except ValueError:
            return {}

    high_k = get_keywords(high_impact_df)
    low_k = get_keywords(low_impact_df)
    
    # Calculate usage rate (%)
    high_n = len(high_impact_df)
    low_n = len(low_impact_df)
    
    # Combine keys
    all_terms = set(high_k.keys()) | set(low_k.keys())
    
    score_data = []
    for term in all_terms:
        h_count = high_k.get(term, 0.001) # smoothing
        l_count = low_k.get(term, 0.001)
        
        h_pct = (h_count / high_n) * 100
        l_pct = (l_count / low_n) * 100
        
        # Frequency Filter: Ignore terms that are extremely rare (< 2 occurrences total)
        if (h_count + l_count) < 3: 
            continue
            
        # Impact Ratio (How much more likely in High Impact?)
        ratio = h_pct / (l_pct + 0.01)
        
        score_data.append({
            "term": term, 
            "high_pct": h_pct, 
            "low_pct": l_pct, 
            "ratio": ratio,
            "diff": h_pct - l_pct
        })
        
    score_df = pd.DataFrame(score_data)
    
    # --- REPORT GENERATION ---
    report_path = "docs/paper/automated_impact_report.md"
    with open(report_path, "w") as f:
        f.write("# Automated Impact Analysis Report\n\n")
        f.write(f"> **Dataset:** {len(refined_df)} papers (2020-2025) from Semantic Scholar.\n")
        f.write(f"> **High Impact (Top 25%):** > {high_threshold} citations (n={len(high_impact_df)})\n")
        f.write(f"> **Low Impact (Bottom 25%):** < {low_threshold} citations (n={len(low_impact_df)})\n\n")

        # 1. STRUCTURAL CONTRAST
        f.write("## 1. Structural Contrast (High vs Low)\n\n")
        f.write("| Feature | High Impact | Low Impact | Diff | Verdict |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        diff_colon = high_stats['colon_pct'] - low_stats['colon_pct']
        f.write(f"| **Colon Usage (Branding)** | {high_stats['colon_pct']:.1f}% | {low_stats['colon_pct']:.1f}% | {diff_colon:+.1f}% | {'✅ Essential' if diff_colon > 5 else '❌ Avoid'} |\n")
        
        for conn in ['via', 'with', 'for']:
             h = high_stats['connectors'][conn]
             l = low_stats['connectors'][conn]
             diff = h - l
             verdict = "✅ Good" if diff > 5 else "❌ Negative Signal" if diff < -5 else "Neutral"
             f.write(f"| **Connector: '{conn}'** | {h:.1f}% | {l:.1f}% | {diff:+.1f}% | {verdict} |\n")
        
        f.write(f"| **Avg Length** | {high_stats['avg_len']:.1f} | {low_stats['avg_len']:.1f} | {high_stats['avg_len']-low_stats['avg_len']:+.1f} | N/A |\n\n")

        # 2. START STRATEGY
        f.write("## 2. Start Strategy Showdown\n\n")
        f.write("| Strategy | Count | Mean Citations | Median | Verdict |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        for strat, row in strategy_stats.iterrows():
            verdict = "🏆 High Risk/Reward" if row['mean'] > 100 else "🛡️ Standard" if row['mean'] > 60 else "💀 Low Ceiling"
            f.write(f"| **{strat}** | {int(row['count'])} | {row['mean']:.1f} | {row['median']:.1f} | {verdict} |\n")
        f.write("\n")

        # 2.5 POST-COLON ANALYSIS (The "Hybrid" Hypothesis)
        f.write("## 2.5 Post-Colon Strategy (The Second Start)\n\n")
        
        def get_post_colon_start(title):
            if ':' not in str(title): return None
            parts = str(title).split(':', 1)
            if len(parts) < 2: return None
            # Get first word of second part
            second_part = parts[1].strip()
            if not second_part: return None
            return second_part.split()[0].lower()

        high_impact_colon_titles = high_impact_df[high_impact_df['Title'].str.contains(':', na=False)]
        post_colon_words = high_impact_colon_titles['Title'].apply(get_post_colon_start)
        post_counter = collections.Counter(post_colon_words)
        
        f.write("| Post-Colon Start Word | Count | Type | Verdict |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        verbs = ['enhancing', 'improving', 'boosting', 'evaluating', 'comparing', 'learning', 'towards', 'bridging', 'unifying']
        
        for w, c in post_counter.most_common(10):
            w_clean = w.lower()
            w_type = "⚡ VERB" if w_clean in verbs or w_clean.endswith('ing') else "Noun/Adj"
            verdict = "✅ Hybrid Candidate" if w_type == "⚡ VERB" else "Standard"
            f.write(f"| **{w.capitalize()}** | {c} | {w_type} | {verdict} |\n")
        f.write("\n")

        # 3. KEYWORD CONTRAST
        f.write("## 3. Keyword Contrast (Terms of Power)\n\n")
        f.write("| Term | High Impact Freq | Low Impact Freq | Gain/Loss |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        # Power Terms
        f.write("| **⚡ POSITIVE CORRELATION** | | | |\n")
        power_terms = score_df.sort_values('diff', ascending=False).head(8)
        for _, row in power_terms.iterrows():
            f.write(f"| {row['term']} | {row['high_pct']:.1f}% | {row['low_pct']:.1f}% | **{row['diff']:+.1f}%** |\n")

        # Mediocrity Terms
        f.write("| **⚠️ NEGATIVE CORRELATION** | | | |\n")
        mod_terms = score_df.sort_values('diff', ascending=True).head(8)
        for _, row in mod_terms.iterrows():
             f.write(f"| {row['term']} | {row['high_pct']:.1f}% | {row['low_pct']:.1f}% | **{row['diff']:+.1f}%** |\n")

    print(f"Report generated successfully: {report_path}")

if __name__ == "__main__":
    analyze_titles_and_keywords()
