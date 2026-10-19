"""Reusable sample profiling and figures for the Assignment 1 deliverables."""
from collections import Counter
from pathlib import Path
import math
import statistics

import matplotlib.pyplot as plt
import numpy as np

NAVY = '#17324d'
TEAL = '#167d8d'
GREY = '#6d7883'


def setup_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titleweight': 'bold',
        'axes.labelcolor': NAVY,
        'text.color': NAVY,
        'axes.titlecolor': NAVY,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def _percentile(values, percentile):
    """Linear percentile without requiring pandas for sample profiling."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile / 100
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def profile_recipes(recipes, sample_seed=42):
    """Build an EDA profile using only the bundled recipe sample."""
    if not recipes:
        raise ValueError('At least one recipe is required.')
    fields = list(recipes[0])
    ingredient_counts = [len(row['ingredients']) for row in recipes]
    servings = [float(row['servings']) for row in recipes]
    categories = Counter(row.get('category') or 'Missing' for row in recipes)
    years = []
    for row in recipes:
        value = row.get('published_at')
        try:
            years.append(int(str(value)[:4]))
        except (TypeError, ValueError):
            pass

    def top_level_missing(field):
        count = 0
        for row in recipes:
            value = row.get(field)
            if value is None or value == '' or (isinstance(value, list) and not value):
                count += 1
        return count

    mismatch = sum(
        len(row['source_ingredients']) != len(row['source_quantities'])
        for row in recipes
    )
    nested_missing = sum(
        any(value is None for value in row['source_quantities'])
        for row in recipes
    )
    servings_above_20 = sum(value > 20 for value in servings)

    return {
        'source_url': 'https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews',
        'source_file': 'recipes.csv',
        'retrieved_date': '2026-10-17',
        'license_as_stated_by_publisher': 'CC0: Public Domain',
        'rows': len(recipes),
        'fields': fields,
        'field_count': len(fields),
        'sample_seed': sample_seed,
        'sample_filter': (
            'Valid name; 3-20 distinct normalized ingredients; positive known '
            'servings; duplicate normalized ingredient sets removed; reservoir '
            'sampling with seed 42.'
        ),
        'missing': {field: top_level_missing(field) for field in fields},
        'quality_flags': {
            'quantity_ingredient_length_mismatch': mismatch,
            'quantity_vector_has_missing_entry': nested_missing,
            'servings_above_20': servings_above_20,
        },
        'ingredient_count_histogram': dict(sorted(Counter(ingredient_counts).items())),
        'ingredient_count': {
            'minimum': min(ingredient_counts),
            'q1': _percentile(ingredient_counts, 25),
            'median': float(statistics.median(ingredient_counts)),
            'q3': _percentile(ingredient_counts, 75),
            'maximum': max(ingredient_counts),
            'mean': float(statistics.mean(ingredient_counts)),
        },
        'servings': {
            'minimum': min(servings),
            'q1': _percentile(servings, 25),
            'median': float(statistics.median(servings)),
            'q3': _percentile(servings, 75),
            'maximum': max(servings),
            'mean': float(statistics.mean(servings)),
        },
        'top_categories': categories.most_common(10),
        'publication_year_min': min(years) if years else None,
        'publication_year_max': max(years) if years else None,
        'missing_definition': (
            'Top-level None, empty string or empty list. Missing entries inside '
            'source quantity vectors are reported separately as a quality flag.'
        ),
    }


def eda_figures(profile, directory):
    """Create EDA figures that describe only the 8,000-record working sample."""
    setup_style()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = profile['rows']
    flags = profile['quality_flags']

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), layout='constrained')
    labels = ['Serving count > 20', 'Quantity vector has NA', 'Quantity/name length mismatch']
    counts = [
        flags['servings_above_20'],
        flags['quantity_vector_has_missing_entry'],
        flags['quantity_ingredient_length_mismatch'],
    ]
    values = [100 * count / rows for count in counts]
    bars = axes[0].barh(labels, values, color=[GREY, TEAL, NAVY], height=0.62)
    axes[0].bar_label(
        bars,
        labels=[f'{count:,} ({value:.1f}%)' for count, value in zip(counts, values)],
        padding=3,
        fontsize=8.5,
    )
    axes[0].set_xlim(0, 90)
    axes[0].set_xlabel('Share of the 8,000-recipe sample (%)')
    axes[0].set_title('Checks requiring caution', loc='left', pad=12)

    histogram = {int(key): value for key, value in profile['ingredient_count_histogram'].items()}
    x_values = list(range(3, 21))
    axes[1].bar(x_values, [histogram.get(key, 0) for key in x_values], color=TEAL, width=0.8)
    axes[1].set_xticks([3, 5, 8, 10, 15, 20])
    axes[1].set_xlabel('Distinct normalised ingredients per recipe')
    axes[1].set_ylabel('Recipes')
    axes[1].set_title('Ingredient counts in the sample', loc='left', pad=12)
    fig.savefig(directory / 'eda_quality_and_ingredients.png', dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    top = profile['top_categories'][:8]
    remaining = rows - sum(value for _, value in top)
    labels = [key for key, _ in top] + ['All other categories']
    values = [value for _, value in top] + [remaining]
    fig, ax = plt.subplots(figsize=(9.2, 4.4), layout='constrained')
    ax.barh(labels[::-1], values[::-1], color=[GREY] + [TEAL] * 8)
    ax.set_xlabel('Recipes')
    ax.set_title('Recipe categories in the 8,000-recipe sample', loc='left')
    fig.savefig(directory / 'eda_categories.png', dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def benchmark_figure(metrics, directory):
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 3.5), layout='constrained')
    values = 100 * metrics['recovery_at_5'].to_numpy()
    bars = ax.barh(metrics['model'][::-1], values[::-1], color=[GREY, TEAL, NAVY], height=0.58)
    ax.bar_label(bars, labels=[f'{value:.1f}%' for value in values[::-1]], padding=4)
    ax.set_xlim(0, 105)
    ax.set_xlabel('Source recipe recovered in the first 5 results (%)')
    ax.set_title('Initial recipe recovery sanity check', loc='left', pad=14)
    fig.savefig(Path(directory) / 'initial_recipe_recovery.png', dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)
