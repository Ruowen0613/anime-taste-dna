from collections import defaultdict
from collections.abc import Callable

from app.features.anime.schemas import AnimeTag
from app.features.taste_dna.schemas import (
    TasteContribution,
    TasteDimension,
    TasteDnaResponse,
    TasteRating,
)

METHOD_VERSION = "metadata-affinity-v2"
NEUTRAL_RATING = 5.5
RATING_HALF_RANGE = 4.5
PRIOR_STRENGTH = 2.0
MAX_CONTRIBUTIONS = 3
MAX_TAGS_PER_GROUP = 5
GENRE_AXES = (
    "Action",
    "Comedy",
    "Drama",
    "Romance",
    "Fantasy",
    "Sci-Fi",
    "Slice of Life",
    "Psychological",
)
STORY_THEME_CATEGORIES = {
    "Theme-Action",
    "Theme-Comedy",
    "Theme-Drama",
    "Theme-Fantasy",
    "Theme-Other",
    "Theme-Romance",
    "Theme-Sci-Fi",
    "Theme-Sci-Fi-Mecha",
    "Theme-Slice of Life",
}


def build_taste_dna(ratings: list[TasteRating]) -> TasteDnaResponse:
    mapped_ids = {
        rating.mal_id for rating in ratings if rating.genres or _usable_tags(rating.tags)
    }
    genre_dimensions = [
        _build_dimension(
            genre,
            [(rating, 100) for rating in ratings if genre in rating.genres],
        )
        for genre in GENRE_AXES
    ]
    story_themes = _build_tag_group(
        ratings,
        lambda category: category in STORY_THEME_CATEGORIES,
    )
    worlds_and_settings = _build_tag_group(
        ratings,
        lambda category: category.startswith("Setting-"),
    )
    characters = _build_tag_group(
        ratings,
        lambda category: category.startswith("Cast-"),
    )

    unmapped = [rating.mal_id for rating in ratings if rating.mal_id not in mapped_ids]
    if not mapped_ids:
        metadata_status = "unavailable"
    elif len(mapped_ids) == len(ratings):
        metadata_status = "complete"
    else:
        metadata_status = "partial"
    return TasteDnaResponse(
        method_version=METHOD_VERSION,
        mapped_ratings=len(mapped_ids),
        total_ratings=len(ratings),
        metadata_status=metadata_status,
        unmapped_mal_ids=unmapped,
        genre_dimensions=genre_dimensions,
        story_themes=story_themes,
        worlds_and_settings=worlds_and_settings,
        characters=characters,
    )


def _usable_tags(tags: list[AnimeTag]) -> list[AnimeTag]:
    return [
        tag
        for tag in tags
        if not tag.is_adult and not tag.is_general_spoiler and not tag.is_media_spoiler
    ]


def _build_tag_group(
    ratings: list[TasteRating],
    category_matches: Callable[[str], bool],
) -> list[TasteDimension]:
    contributions: dict[str, list[tuple[TasteRating, int]]] = defaultdict(list)
    for rating in ratings:
        for tag in _usable_tags(rating.tags):
            if category_matches(tag.category):
                contributions[tag.name].append((rating, tag.rank))

    dimensions = [
        _build_dimension(label, items) for label, items in contributions.items()
    ]
    dimensions.sort(
        key=lambda dimension: (-dimension.score, -dimension.support, dimension.label)
    )
    return dimensions[:MAX_TAGS_PER_GROUP]


def _build_dimension(
    label: str,
    weighted_ratings: list[tuple[TasteRating, int]],
) -> TasteDimension:
    if not weighted_ratings:
        return TasteDimension(
            label=label,
            score=50.0,
            support=0,
            average_rating=None,
            contributing_titles=[],
        )

    weighted_centered_sum = sum(
        ((rating.rating - NEUTRAL_RATING) / RATING_HALF_RANGE) * (rank / 100)
        for rating, rank in weighted_ratings
    )
    total_weight = sum(rank / 100 for _, rank in weighted_ratings)
    shrunk_affinity = weighted_centered_sum / (total_weight + PRIOR_STRENGTH)
    score = round(50.0 + 50.0 * shrunk_affinity, 1)
    average_rating = round(
        sum(rating.rating for rating, _ in weighted_ratings) / len(weighted_ratings), 1
    )
    ranked_contributions = sorted(
        weighted_ratings,
        key=lambda item: (
            -abs(item[0].rating - NEUTRAL_RATING) * item[1],
            -item[0].rating,
            item[0].title,
        ),
    )
    return TasteDimension(
        label=label,
        score=score,
        support=len(weighted_ratings),
        average_rating=average_rating,
        contributing_titles=[
            TasteContribution(
                mal_id=rating.mal_id,
                title=rating.title,
                rating=rating.rating,
            )
            for rating, _ in ranked_contributions[:MAX_CONTRIBUTIONS]
        ],
    )
