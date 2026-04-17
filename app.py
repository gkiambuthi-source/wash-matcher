import os
import urllib.request
import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ---------- Configuration ----------
CSV_URL = "https://github.com/openwashdata/washopenresearch/raw/main/inst/extdata/washdev.csv"
CSV_FILE = "washdev.csv"
GROQ_MODEL = "llama-3.1-8b-instant"   # updated model
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Optional: scrape RWSN (set to True to enable)
SCRAPE_RWSN = False

app = Flask(__name__)

# ---------- Initialize Groq ----------
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None
    print("WARNING: GROQ_API_KEY not set. LLM summaries disabled.")

# ---------- Load knowledge base (with URLs) ----------
def load_knowledge_base():
    """Download CSV if needed and return dict: title -> {description, url}."""
    if not os.path.exists(CSV_FILE):
        print("Downloading washdev.csv...")
        urllib.request.urlretrieve(CSV_URL, CSV_FILE)

    df = pd.read_csv(CSV_FILE)
    df['description'] = df['title'].fillna('') + " " + df['keywords'].fillna('')
    df = df.dropna(subset=['title'])

    products = {}
    for _, row in df.iterrows():
        title = row['title']
        # The CSV column containing the URL is named 'publication url'
        url = row.get('paper_url', '')
        if pd.isna(url):
            url = ''
        products[title] = {
            'description': row['description'],
            'url': url
        }

    print(f"Loaded {len(products)} products from CSV.")
    return products

# ---------- Optional RWSN scraper (with URLs) ----------
def scrape_rwsn():
    """Scrape titles, descriptions, and URLs from RWSN library."""
    import requests
    from bs4 import BeautifulSoup
    import time

    url = "https://rural-water-supply.net/en/resources"
    print("🌐 Scraping RWSN library...")
    try:
        time.sleep(1)
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')
        resources = {}
        for item in soup.find_all('div', class_='list-group-item'):
            a_tag = item.find('a')
            p_tag = item.find('p')
            if a_tag and p_tag:
                title = a_tag.get_text(strip=True)
                link = a_tag.get('href')
                if link and not link.startswith('http'):
                    link = "https://rural-water-supply.net" + link
                desc = p_tag.get_text(strip=True)
                resources[title] = {'description': desc, 'url': link}
        print(f"   Scraped {len(resources)} resources from RWSN.")
        return resources
    except Exception as e:
        print(f"   RWSN scraping failed: {e}")
        return {}

# ---------- Matching function (returns titles, scores, urls) ----------
def match_query(query, products_dict, top_n=5):
    if not query:
        return []
    titles = list(products_dict.keys())
    descriptions = [products_dict[title]['description'] for title in titles]

    vectorizer = TfidfVectorizer(stop_words="english")
    all_texts = [query] + descriptions
    tfidf = vectorizer.fit_transform(all_texts)
    query_vec = tfidf[0:1]
    doc_vecs = tfidf[1:]
    scores = cosine_similarity(query_vec, doc_vecs).flatten()
    top_indices = scores.argsort()[-top_n:][::-1]

    results = []
    for i in top_indices:
        if scores[i] > 0.05:
            title = titles[i]
            url = products_dict[title]['url']
            results.append((title, scores[i], url))
    return results

# ---------- LLM summarization ----------
def summarize_with_groq(query, product_title, product_desc, score):
    if not groq_client:
        return "LLM summaries not available (missing API key)."
    prompt = f"""You are a WASH expert.
User asked: "{query}"
Product: "{product_title}"
Description: {product_desc}
Relevance score: {score:.2f}

Explain in 2-3 sentences why this product is useful, for who and how to use it."""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary error: {str(e)[:100]}"

# ---------- Flask routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        unit = request.form.get('unit')
        custom_query = request.form.get('query')
        query = custom_query.strip() if custom_query else unit
        if not query:
            return render_template('index.html', error="Please enter a query.",
                                   units=["Water Quality", "Leakage Reduction", "Customer Service", "Wastewater Treatment"])

        try:
            products = load_knowledge_base()
            # Optional: add scraped data
            if SCRAPE_RWSN:
                scraped = scrape_rwsn()
                for title, data in scraped.items():
                    if title not in products:
                        products[title] = data

            matches = match_query(query, products, top_n=3)
            summaries = []
            for title, score, url in matches:
                desc = products[title]['description']
                summary = summarize_with_groq(query, title, desc, score)
                summaries.append((title, score, summary, url))
            return render_template('results.html', query=query, matches=summaries)
        except Exception as e:
            error_msg = f"Internal error: {str(e)[:200]}"
            return render_template('index.html', error=error_msg, units=[])

    return render_template('index.html', units=["Water Quality", "Leakage Reduction", "Customer Service", "Wastewater Treatment"])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
