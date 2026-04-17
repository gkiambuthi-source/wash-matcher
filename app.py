import os
import pickle
import urllib.request
import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from bs4 import BeautifulSoup
import requests
import time

# ---------- Configuration ----------
CSV_URL = "https://github.com/openwashdata/washopenresearch/raw/main/inst/extdata/washdev.csv"
CSV_FILE = "washdev.csv"
PRODUCTS_PKL = "knowledge_base/products.pkl"
GROQ_MODEL = "llama3-8b-8192"   # free tier model
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")   # set this environment variable

# Optional: scrape RWSN? Set to True to enable live scraping (slower)
SCRAPE_RWSN = True

app = Flask(__name__)

# ---------- Initialize Groq client ----------
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None
    print("⚠️ GROQ_API_KEY not set. LLM summaries will be disabled.")

# ---------- Knowledge base loading (CSV + optional RWSN scraping) ----------
def download_csv():
    if not os.path.exists(CSV_FILE):
        print("📥 Downloading washdev.csv...")
        urllib.request.urlretrieve(CSV_URL, CSV_FILE)

def scrape_rwsn():
    """Scrape titles & descriptions from RWSN library first page."""
    url = "https://rural-water-supply.net/en/resources"
    print("🌐 Scraping RWSN library...")
    try:
        time.sleep(1)
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')
        resources = {}
        # Try common selectors
        for item in soup.find_all('div', class_='list-group-item'):
            a = item.find('a')
            p = item.find('p')
            if a and p:
                title = a.get_text(strip=True)
                desc = p.get_text(strip=True)
                resources[title] = desc
        print(f"   Scraped {len(resources)} resources from RWSN.")
        return resources
    except Exception as e:
        print(f"   RWSN scraping failed: {e}")
        return {}

def load_knowledge_base():
    """Load or create the knowledge base dictionary {title: description}."""
    if os.path.exists(PRODUCTS_PKL):
        with open(PRODUCTS_PKL, "rb") as f:
            return pickle.load(f)
    
    # Build from CSV
    download_csv()
    df = pd.read_csv(CSV_FILE)
    df['description'] = df['title'].fillna('') + " " + df['keywords'].fillna('')
    df = df.dropna(subset=['title'])
    products = dict(zip(df['title'], df['description']))
    
    # Optionally add scraped RWSN data
    if SCRAPE_RWSN:
        rwsn_products = scrape_rwsn()
        for title, desc in rwsn_products.items():
            if title not in products:
                products[title] = desc
    
    # Cache for next runs
    os.makedirs(os.path.dirname(PRODUCTS_PKL), exist_ok=True)
    with open(PRODUCTS_PKL, "wb") as f:
        pickle.dump(products, f)
    print(f"📚 Knowledge base ready: {len(products)} products.")
    return products

# ---------- Matching function ----------
def match_query(query, products_dict, top_n=5):
    """Return list of (title, score) for top matches."""
    if not query:
        return []
    titles = list(products_dict.keys())
    descriptions = list(products_dict.values())
    vectorizer = TfidfVectorizer(stop_words="english")
    all_texts = [query] + descriptions
    tfidf = vectorizer.fit_transform(all_texts)
    query_vec = tfidf[0:1]
    doc_vecs = tfidf[1:]
    scores = cosine_similarity(query_vec, doc_vecs).flatten()
    top_indices = scores.argsort()[-top_n:][::-1]
    results = [(titles[i], scores[i]) for i in top_indices if scores[i] > 0.05]
    return results

# ---------- LLM summarization using Groq ----------
def summarize_with_groq(query, product_title, product_desc, score):
    if not groq_client:
        return "LLM summaries not available (missing API key)."
    
    prompt = f"""You are a WASH (Water, Sanitation and Hygiene) expert advisor.
User asked: "{query}"
A knowledge product titled "{product_title}" has this description: {product_desc}
The relevance score is {score:.2f} (0=unrelated, 1=perfect match).

Write one short paragraph (2-4 sentences) explaining:
1. Why this product is relevant to the user's question.
2. How the user could concretely use this knowledge in their work.
Be practical, specific, and avoid generic statements."""
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=180
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq error: {e}]"

# ---------- Flask routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        unit = request.form.get('unit')
        custom_query = request.form.get('query')
        if custom_query and custom_query.strip():
            query = custom_query.strip()
        else:
            query = unit
        if not query:
            return render_template('index.html', error="Please enter a query or select a unit.",
                                   units=["Water Quality", "Leakage Reduction", "Customer Service", "Wastewater Treatment"])
        
        products = load_knowledge_base()
        matches = match_query(query, products, top_n=3)
        
        summaries = []
        for title, score in matches:
            desc = products[title]
            summary = summarize_with_groq(query, title, desc, score)
            summaries.append((title, score, summary))
        
        return render_template('results.html', query=query, matches=summaries)
    
    return render_template('index.html', units=[
        "Water Quality", "Leakage Reduction", "Customer Service", "Wastewater Treatment"
    ])

if __name__ == '__main__':
    app.run(debug=True, port=5000)