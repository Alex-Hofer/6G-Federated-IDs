"""Configuration loading with environment variable interpolation and validation.

Provides YAML configuration loading for the federated IDS system. Supports
``${VAR_NAME:-default}`` syntax in YAML values for environment variable
interpolation, enabling portability across different machines and clusters
without modifying the config file.

Example:
    >>> from federated_ids.config import load_config
    >>> config = load_config("config/default.yaml")
    >>> print(config["seed"])
    42
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml

# Pattern for ${VAR_NAME:-default_value} or ${VAR_NAME}
_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")

# Required top-level config sections and their expected types
_REQUIRED_SECTIONS = {
    "data": dict,
    "model": dict,
    "training": dict,
    "federation": dict,
    "seed": int,
}

# Valid values for the log_level configuration key
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Required nested keys per section
_REQUIRED_NESTED = {
    "data": ["raw_dir", "processed_dir", "files", "test_size"],
    "model": ["hidden_layers", "dropout", "num_classes"],
    "training": ["learning_rate", "local_epochs", "batch_size"],
    "federation": ["num_clients", "num_rounds", "fraction_fit"],
}


def _env_constructor(loader: yaml.Loader, node: yaml.Node) -> str:
    """Resolve ``${VAR_NAME:-default}`` patterns in YAML string values.

    Args:
        loader: The YAML loader instance.
        node: The YAML node containing the string value.

    Returns:
        The string with all environment variable references resolved.
        If a variable is set in the environment, its value is used.
        Otherwise, the default value after ``:-`` is used. If no default
        is specified, an empty string is substituted.
    """
    value = loader.construct_scalar(node)

    def _replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)

    return _ENV_PATTERN.sub(_replacer, value)


def validate_log_level(level: str) -> str:
    """Validate that a log level string is one of the standard Python levels.

    Args:
        level: Log level name (e.g. ``"INFO"``, ``"DEBUG"``).

    Returns:
        The validated log level string (unchanged).

    Raises:
        ValueError: If ``level`` is not a valid Python log level.
    """
    if level not in _VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log_level '{level}'. Must be one of: {sorted(_VALID_LOG_LEVELS)}"
        )
    return level


def validate_config_path(path: str) -> str:
    """Validate that a configuration file path has a YAML extension.

    Args:
        path: Path to the configuration file.

    Returns:
        The validated path string (unchanged).

    Raises:
        ValueError: If ``path`` does not end with ``.yaml`` or ``.yml``.
    """
    if not path.endswith((".yaml", ".yml")):
        raise ValueError(
            f"Config path '{path}' must have .yaml or .yml extension"
        )
    return path


# Custom YAML loader that supports env var interpolation.
# IMPORTANT: Subclass SafeLoader to avoid modifying the global SafeLoader
# class, which would affect all YAML loading in the process (Pitfall 7).
class EnvYamlLoader(yaml.SafeLoader):
    """YAML loader with environment variable interpolation support."""

    pass


EnvYamlLoader.add_implicit_resolver("!env", _ENV_PATTERN, None)
EnvYamlLoader.add_constructor("!env", _env_constructor)


def _validate_config(config: dict[str, Any]) -> None:
    """Validate that all required configuration sections and keys are present.

    Args:
        config: The parsed configuration dictionary.

    Raises:
        ValueError: If required top-level sections are missing, have wrong
            types, or are missing required nested keys. The error message
            lists all missing/invalid items for fast debugging.
    """
    errors = []

    # Check required top-level sections
    for section, expected_type in _REQUIRED_SECTIONS.items():
        if section not in config:
            errors.append(f"Missing required section: '{section}'")
        elif not isinstance(config[section], expected_type):
            errors.append(
                f"Section '{section}' must be {expected_type.__name__}, "
                f"got {type(config[section]).__name__}"
            )

    # Check required nested keys (only if section exists and is a dict)
    for section, keys in _REQUIRED_NESTED.items():
        if section in config and isinstance(config[section], dict):
            for key in keys:
                if key not in config[section]:
                    errors.append(f"Missing required key: '{section}.{key}'")

    # Validate log_level if present
    if "log_level" in config:
        try:
            validate_log_level(config["log_level"])
        except ValueError as e:
            errors.append(str(e))

    if errors:
        error_list = "\n  - ".join(errors)
        raise ValueError(f"Configuration validation failed:\n  - {error_list}")


def load_config(path: str) -> dict[str, Any]:
    """Load and validate a YAML configuration file with env var interpolation.

    Loads the specified YAML file, resolves any ``${VAR:-default}`` environment
    variable references in string values, and validates that all required
    sections and keys are present.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        The parsed and validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required configuration sections or keys are missing.

    Example:
        >>> config = load_config("config/default.yaml")
        >>> config["data"]["test_size"]
        0.2
    """
    validate_config_path(path)
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(config_path) as f:
        config = yaml.load(f, Loader=EnvYamlLoader)

    _validate_config(config)
    return config
