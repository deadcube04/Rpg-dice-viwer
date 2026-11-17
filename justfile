set windows-shell := ["cmd.exe", "/D", "/C"]

python := env_var_or_default("PYTHON", "python")
release := "v1-initial-20261117-r3"

default:
    @just --list

# Restore and verify the approved model.
restore:
    {{python}} tools/model.py restore

# Start CPU serving.
up project="dice-model" port="23000":
    {{python}} tools/model.py run --project "{{project}}" --port "{{port}}"

cpu project="dice-model" port="23000":
    {{python}} tools/model.py run --project "{{project}}" --port "{{port}}"

# Start NVIDIA/CUDA serving.
cuda project="dice-model" port="23000":
    {{python}} tools/model.py run --cuda --project "{{project}}" --port "{{port}}"

# Check HTTP readiness.
health port="23000":
    curl --include --fail --silent --show-error --max-time 10 "http://127.0.0.1:{{port}}/readyz"

# Submit an image; use image/jpeg for JPEG files.
recognize image port="23000" mime="image/png":
    curl --silent --show-error --include --max-time 45 -X POST "http://127.0.0.1:{{port}}/v1/recognize" -H "Content-Type: {{mime}}" --data-binary "@{{image}}"

status project="dice-model":
    docker compose --project-name "{{project}}" --project-directory . --env-file "runtime/portable/{{project}}/releases/{{release}}.env" -f "releases/{{release}}/deploy/v1/compose.yaml" ps --all

logs project="dice-model":
    docker compose --project-name "{{project}}" --project-directory . --env-file "runtime/portable/{{project}}/releases/{{release}}.env" -f "releases/{{release}}/deploy/v1/compose.yaml" logs --tail 100 api

# Stop services while preserving volumes.
stop project="dice-model":
    docker compose --project-name "{{project}}" --project-directory . --env-file "runtime/portable/{{project}}/releases/{{release}}.env" -f "releases/{{release}}/deploy/v1/compose.yaml" stop

model-info:
    {{python}} -m json.tool "models/{{release}}/manifest.json"
