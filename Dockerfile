FROM python:3.11.5-slim AS builder

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

# to run poetry directly as soon as it's installed
ENV PATH="$POETRY_HOME/bin:$PATH"

# install poetry
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# copy only pyproject.toml and poetry.lock file nothing else here
COPY poetry.lock pyproject.toml ./

# this will create the folder /app/.venv (might need adjustment depending on which poetry version you are using)
RUN poetry lock && poetry install --no-root --no-ansi --without dev

# ---------------------------------------------------------------------

FROM python:3.11.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# copy the venv folder from builder image
COPY --from=builder /app/.venv ./.venv

# copy the application code
COPY . .

EXPOSE 8000
# Run main.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
