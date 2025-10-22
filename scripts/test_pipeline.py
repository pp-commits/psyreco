# scripts/test_pipeline.py
from modules.llm_analyzer import analyze_mood_mistral
from modules.recommender import recommend_books
import json

if __name__ == "__main__":
    text = "I feel burned out and anxious about my career and need clarity and calm. I'd like something practical."
    print("Input:", text)
    tags = analyze_mood_mistral(text)
    print("Tags:", json.dumps(tags, indent=2))
    recs = recommend_books(tags, n=5)
    print("Results (titles):")
    for md in recs.get("metadatas", [[]])[0]:
        print("-", md.get("title"), "by", md.get("author"))
