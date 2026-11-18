from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.features.anime.client import close_anilist_client
from app.features.anime.router import router as anime_router
from app.features.popular_picks.router import router as popular_picks_router
from app.features.preference_quiz.router import router as preference_quiz_router
from app.features.recommendations.router import router as recommendations_router
from app.features.system.router import router as system_router
from app.features.taste_dna.router import router as taste_dna_router

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await close_anilist_client()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Online inference and metadata API for Anime Taste DNA.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(system_router)
app.include_router(anime_router)
app.include_router(popular_picks_router)
app.include_router(recommendations_router)
app.include_router(preference_quiz_router)
app.include_router(taste_dna_router)
