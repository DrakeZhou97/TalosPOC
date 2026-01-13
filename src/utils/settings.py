"""Centralized application settings."""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class ChatModelConfig(BaseModel):
    """Configuration required to instantiate a chat model."""

    model: str = "gpt-5.2"
    # model: str = "qwen3-vl-30b-a3b-instruct"
    # base_url: str = "https://yunwu.ai/v1/"
    base_url: str = "https://api.openai.com/v1/"
    # api_key: str = "sk-C593NzPP7sczeVcuS89YBb6Ejkw6gofusjXJpv1QDHMln5QM"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    timeout: float | None = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")


class AgentModelSettings(BaseModel):
    """Container for every agent specific model configuration."""

    watchdog: ChatModelConfig = Field(
        default_factory=lambda: ChatModelConfig(
            # model="gpt-5-mini",
            temperature=0,
            # top_p=0.95,
            seed=42,
        ),
    )
    intention_detection: ChatModelConfig = Field(
        default_factory=lambda: ChatModelConfig(
            # model="gpt-5-mini",
            temperature=0,
            # top_p=0.95,
            seed=42,
        ),
    )

    tlc_agent: ChatModelConfig = Field(
        default_factory=lambda: ChatModelConfig(
            # model="gpt-5-mini",
            temperature=0,
            # top_p=0.95,
            seed=42,
        ),
    )
    
    cc_agent: ChatModelConfig = Field(
        default_factory=lambda: ChatModelConfig(
            # model="gpt-5-mini",
            temperature=0,
            # top_p=0.95,
            seed=42,
        ),
    )

    re_agent: ChatModelConfig = Field(
        default_factory=lambda: ChatModelConfig(
            # model="gpt-5-mini",
            temperature=0,
            # top_p=0.95,
            seed=42,
        ),
    )

    planner: ChatModelConfig = Field(
        default_factory=lambda: ChatModelConfig(
            # model="gpt-5-mini",
            temperature=0,
            # top_p=0.95,
            seed=42,
        ),
    )


class Settings(BaseSettings):
    """Application level settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    agents: AgentModelSettings = Field(default_factory=AgentModelSettings)


settings = Settings()

__all__ = ["AgentModelSettings", "ChatModelConfig", "settings"]
