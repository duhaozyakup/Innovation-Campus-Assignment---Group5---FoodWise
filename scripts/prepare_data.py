"""Create the reproducible 8,000-record FoodWise analysis dataset."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis import profile_recipes
from src.foodwise import normalize_ingredient, parse_r_vector


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def prepare(source_zip: Path, sample_size=8000, seed=42):
    """Filter the source stream, reservoir-sample it and profile that sample."""
    rng = random.Random(seed)
    sample = []
    seen = set()
    selection = Counter()
    csv.field_size_limit(10_000_000)

    with ZipFile(source_zip) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith('.csv'))
        with archive.open(csv_name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''))
            original_columns = reader.fieldnames
            for row in reader:
                selection['source_rows_scanned'] += 1
                ingredients = [
                    normalize_ingredient(value)
                    for value in parse_r_vector(row['RecipeIngredientParts'])
                    if value and str(value).strip()
                ]
                servings = number(row.get('RecipeServings'))
                canonical = sorted(set(ingredients))
                if not row.get('Name') or not (3 <= len(canonical) <= 20) or servings is None or servings <= 0:
                    selection['excluded_before_deduplication'] += 1
                    continue

                key = hashlib.sha256('\x1f'.join(canonical).encode()).digest()
                if key in seen:
                    selection['duplicate_ingredient_sets_after_filter'] += 1
                    continue
                seen.add(key)
                selection['eligible_unique_rows'] += 1
                eligible_number = selection['eligible_unique_rows']

                if len(sample) < sample_size:
                    slot = len(sample)
                else:
                    slot = rng.randrange(eligible_number)
                    if slot >= sample_size:
                        continue

                recipe = {
                    'recipe_id': str(row['RecipeId']),
                    'name': row['Name'],
                    'category': row.get('RecipeCategory'),
                    'ingredients': canonical,
                    'source_ingredients': ingredients,
                    'source_quantities': parse_r_vector(row['RecipeIngredientQuantities']),
                    'servings': servings,
                    'instructions': parse_r_vector(row['RecipeInstructions']),
                    'calories_source': number(row.get('Calories')),
                    'protein_source': number(row.get('ProteinContent')),
                    'published_at': row.get('DatePublished'),
                }
                if slot == len(sample):
                    sample.append(recipe)
                else:
                    sample[slot] = recipe

    sample.sort(key=lambda recipe: int(recipe['recipe_id']))
    profile = profile_recipes(sample, sample_seed=seed)
    metadata = {
        'source_url': profile['source_url'],
        'source_file': csv_name,
        'retrieved_date': profile['retrieved_date'],
        'license_as_stated_by_publisher': profile['license_as_stated_by_publisher'],
        'original_field_count': len(original_columns),
        'source_rows_scanned_for_sampling': selection['source_rows_scanned'],
        'selection_counts': dict(selection),
        'analysis_sample_rows': len(sample),
        'sample_seed': seed,
        'note': 'The full source is used only to select the reproducible sample. EDA and models use the 8,000-record sample.',
    }

    (ROOT / 'data').mkdir(exist_ok=True)
    with (ROOT / 'data/recipes_sample.jsonl').open('w', encoding='utf-8') as output:
        for recipe in sample:
            output.write(json.dumps(recipe, ensure_ascii=False) + '\n')
    (ROOT / 'data/sample_profile.json').write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (ROOT / 'data/source_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps({
        'analysis_sample_rows': len(sample),
        'sample_seed': seed,
        'ingredient_count_median': profile['ingredient_count']['median'],
        'quality_flags': profile['quality_flags'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-zip', required=True, type=Path)
    parser.add_argument('--sample-size', default=8000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    arguments = parser.parse_args()
    prepare(arguments.source_zip, arguments.sample_size, arguments.seed)
