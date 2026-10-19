"""Assignment 1 ingredient retrieval experiments. No food-safety inference."""
from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def parse_r_vector(value: str) -> list:
    """Read the source's c(\"a\", \"b\", NA) lists without executing source text."""
    value = str(value).strip()
    if value in ('', 'NA', 'N/A', 'NaN', 'null', 'c()'):
        return []
    text = value[2:-1].strip() if value.startswith('c(') and value.endswith(')') else value
    if not text:
        return []
    token_pattern = re.compile(r'\s*("(?:\\.|[^"\\])*"|NA|[^,]+)\s*(?:,|$)', re.S)
    tokens = []
    position = 0
    while position < len(text):
        match = token_pattern.match(text, position)
        if match is None or match.end() == position:
            raise ValueError('Unsupported source list format')
        tokens.append(match.group(1))
        position = match.end()
    result = []
    for token in tokens:
        token = token.strip()
        if token == 'NA':
            result.append(None)
        elif token.startswith('"') and token.endswith('"'):
            try:
                result.append(json.loads(token))
            except json.JSONDecodeError:
                result.append(token[1:-1].replace('\\"', '"').replace('\\n', '\n'))
        elif token:
            result.append(token)
    return result


def normalize_ingredient(text: str) -> str:
    text = unicodedata.normalize('NFKC', str(text)).casefold().strip()
    return re.sub(r'\s+', ' ', text)


def identity_tokens(items):
    return items


def load_recipes(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


class RecipeRetriever:
    """Learn ingredient IDF weights; optionally compare a 64-dimensional LSA model."""

    def __init__(self, recipes: list[dict]):
        self.recipes = recipes
        self.ingredients = [set(r['ingredients']) for r in recipes]
        self.vectorizer = TfidfVectorizer(
            analyzer=identity_tokens, lowercase=False, norm='l2', dtype=np.float64
        )
        self.matrix = self.vectorizer.fit_transform([r['ingredients'] for r in recipes])
        self.svd = None
        self.latent = None

    def fit_lsa(self, dimensions=64):
        dimensions = min(dimensions, self.matrix.shape[0] - 1, self.matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=dimensions, random_state=42, n_iter=5)
        self.latent = normalize(self.svd.fit_transform(self.matrix))
        return self

    def tfidf_scores(self, ingredients):
        query = self.vectorizer.transform([[normalize_ingredient(x) for x in ingredients]])
        return (query @ self.matrix.T).toarray().ravel()

    def lsa_scores(self, ingredients):
        if self.svd is None:
            raise RuntimeError('Call fit_lsa before requesting LSA scores.')
        query = self.vectorizer.transform([[normalize_ingredient(x) for x in ingredients]])
        query = normalize(self.svd.transform(query))
        return (query @ self.latent.T).ravel()

    def jaccard_scores(self, ingredients):
        pantry = {normalize_ingredient(x) for x in ingredients}
        return np.array([
            len(pantry & values) / max(1, len(pantry | values)) for values in self.ingredients
        ])

    def recommend(self, inventory: list[dict], as_of: str, top_k=5) -> pd.DataFrame:
        """Rank using entered dates. Quantity and nutrient feasibility are not inferred."""
        today = date.fromisoformat(as_of)
        available = {}
        for item in inventory:
            days = (date.fromisoformat(item['use_by']) - today).days
            if days >= 0 and item.get('usable', True):
                key = normalize_ingredient(item['ingredient'])
                available[key] = min(days, available.get(key, days))
        known = set(available) & set(self.vectorizer.vocabulary_)
        if not known:
            return pd.DataFrame(columns=['recipe_id', 'name', 'score', 'ingredient_coverage', 'urgent_coverage', 'missing_ingredients'])
        cosine = self.tfidf_scores(known)
        urgent = {k for k, days in available.items() if days <= 2}
        rows = []
        for i, recipe in enumerate(self.recipes):
            ingredients = self.ingredients[i]
            matched = ingredients & known
            if not matched:
                continue
            coverage = len(matched) / len(ingredients)
            urgent_coverage = len(ingredients & urgent) / max(1, len(urgent))
            # Initial, explicitly heuristic weights. These have not been optimized.
            score = 0.60 * cosine[i] + 0.30 * coverage + 0.10 * urgent_coverage
            rows.append({
                'recipe_id': recipe['recipe_id'], 'name': recipe['name'],
                'all_ingredient_types_available': len(ingredients - known) == 0,
                'score': float(score), 'tfidf_cosine': float(cosine[i]),
                'ingredient_coverage': coverage, 'urgent_coverage': urgent_coverage,
                'urgent_ingredients_used': sorted(ingredients & urgent),
                'missing_ingredients': sorted(ingredients - known),
                'source_servings': recipe['servings'],
            })
        return pd.DataFrame(rows).sort_values(['all_ingredient_types_available', 'score', 'recipe_id'], ascending=[False, False, True]).head(top_k).reset_index(drop=True)


def recovery_benchmark(engine: RecipeRetriever, n_queries=200, seed=73):
    """Synthetic source-recipe recovery, not independent user relevance or waste impact."""
    rng = np.random.default_rng(seed)
    eligible = [i for i, r in enumerate(engine.recipes) if 5 <= len(r['ingredients']) <= 15]
    targets = rng.choice(eligible, size=min(n_queries, len(eligible)), replace=False)
    detail = []
    for target in targets:
        terms = engine.recipes[int(target)]['ingredients']
        keep = min(6, max(3, math.ceil(0.60 * len(terms))))
        query = [terms[int(j)] for j in sorted(rng.choice(len(terms), size=keep, replace=False))]
        scores_by_model = {
            'Ingredient overlap': engine.jaccard_scores(query),
            'TF-IDF cosine': engine.tfidf_scores(query),
            'LSA 64 dimensions': engine.lsa_scores(query),
        }
        record = {'source_recipe_id': engine.recipes[int(target)]['recipe_id'], 'query': query}
        for name, scores in scores_by_model.items():
            order = np.argsort(-scores, kind='stable')
            rank = int(np.flatnonzero(order == target)[0]) + 1
            detail.append({'model': name, 'source_recipe_id': record['source_recipe_id'], 'query': query, 'rank': rank})
    records = pd.DataFrame(detail)
    summary = []
    for name in ['Ingredient overlap', 'TF-IDF cosine', 'LSA 64 dimensions']:
        ranks = records.loc[records['model'] == name, 'rank']
        summary.append({'model': name, 'queries': len(ranks), 'recovery_at_1': float((ranks <= 1).mean()), 'recovery_at_5': float((ranks <= 5).mean()), 'recovery_at_10': float((ranks <= 10).mean()), 'mean_reciprocal_rank': float((1 / ranks).mean())})
    return pd.DataFrame(summary), detail
