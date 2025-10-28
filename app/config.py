import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _int_from_env(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class Settings:
    base_dir: Path = Path(__file__).resolve().parent.parent
    model_filename: str = field(default_factory=lambda: os.getenv("MODEL_FILENAME", "lgbm_anomaly_multi.pkl"))
    metadata_filename: str = field(default_factory=lambda: os.getenv("MODEL_METADATA_FILENAME", "lgbm_anomaly_meta.pkl"))
    template_dir_env: Optional[str] = field(default_factory=lambda: os.getenv("TEMPLATE_DIR"))
    preview_rows: int = field(default_factory=lambda: _int_from_env("PREVIEW_ROWS", 10))
    top_feature_badge_count: int = field(default_factory=lambda: _int_from_env("TOP_FEATURE_BADGE_COUNT", 10))
    top_feature_list_count: int = field(default_factory=lambda: _int_from_env("TOP_FEATURE_LIST_COUNT", 15))
    supabase_url: Optional[str] = field(default_factory=lambda: os.getenv("SUPABASE_URL"))
    supabase_service_key: Optional[str] = field(
        default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_API_KEY")
    )
    supabase_bucket: str = field(default_factory=lambda: os.getenv("SUPABASE_BUCKET", "anomaly-uploads"))
    supabase_table: str = field(default_factory=lambda: os.getenv("SUPABASE_TABLE", "anomaly.upload_batches"))
    database_url: Optional[str] = field(default_factory=lambda: os.getenv("DATABASE_URL"))
    model_dir: Path = field(init=False)
    model_path: Path = field(init=False)
    metadata_path: Path = field(init=False)
    templates_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.model_dir = Path(os.getenv("MODEL_DIR", self.base_dir / "models"))
        self.model_path = Path(os.getenv("MODEL_PATH", self.model_dir / self.model_filename))
        self.metadata_path = Path(os.getenv("MODEL_METADATA_PATH", self.model_dir / self.metadata_filename))
        template_dir = self.template_dir_env or str(self.base_dir / "templates")
        self.templates_dir = Path(template_dir)

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_key)

    @property
    def database_enabled(self) -> bool:
        return bool(self.database_url)
