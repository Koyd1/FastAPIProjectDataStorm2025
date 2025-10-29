from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .state import AppStore
from .model import load_model_context
from .routes import register_routes
from .supabase_service import (
    fetch_supabase_history,
    init_supabase_client,
    sanitize_dataframe_for_storage,
    update_aggregated_state,
)
from .db_service import DatabaseService

from contextlib import asynccontextmanager


def create_app() -> FastAPI:

    settings = Settings()

    # Creating lifespan manager
    @asynccontextmanager
    async def lifespanmgr(app: FastAPI):
    # startup

        context = load_model_context(settings)
        store.set_context(context)
        store.current_metadata = store.generate_metadata(None)
        store.supabase_client = init_supabase_client(settings)
        if settings.database_enabled:
            db_service = DatabaseService(settings.database_url)
            db_service.init()
            store.db_service = db_service
        else:
            store.db_service = None
        if store.supabase_client is not None:
            history_df = fetch_supabase_history(store)
            if not history_df.empty:
                update_aggregated_state(store, sanitize_dataframe_for_storage(history_df))


        yield

        # shutdown

        if store.db_service is not None:
            store.db_service.close()


        ##### end lifespanmgr




    app = FastAPI(title="Anomaly Analyzer", lifespan=lifespanmgr)
    templates = Jinja2Templates(directory=str(settings.templates_dir))
    static_dir = settings.base_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    store = AppStore(settings=settings)

    # Ensure Supabase module logs are visible with default uvicorn logging.
    supabase_logger = logging.getLogger("app.supabase_service")
    supabase_logger.setLevel(logging.INFO)
    supabase_logger.propagate = True

    app.state.settings = settings
    app.state.templates = templates

    app.state.store = store

    # @app.on_event("startup")
    # async def on_startup() -> None:
    #     context = load_model_context(settings)
    #     store.set_context(context)
    #     store.current_metadata = store.generate_metadata(None)
    #     store.supabase_client = init_supabase_client(settings)
    #     if settings.database_enabled:
    #         db_service = DatabaseService(settings.database_url)
    #         db_service.init()
    #         store.db_service = db_service
    #     else:
    #         store.db_service = None
    #     if store.supabase_client is not None:
    #         history_df = fetch_supabase_history(store)
    #         if not history_df.empty:
    #             update_aggregated_state(store, sanitize_dataframe_for_storage(history_df))

    # @app.on_event("shutdown")
    # async def on_shutdown() -> None:
    #     if store.db_service is not None:
    #         store.db_service.close()

#**************************************************




    register_routes(app)
    return app
