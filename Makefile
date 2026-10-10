.PHONY: install test lint smoke pipeline run drift serve ui up
install:   ; pip install -r requirements-dev.txt
test:      ; pytest -q
lint:      ; ruff check .
smoke:     ; DATA_SOURCE=synthetic python -m src.pipeline      # no network needed
pipeline:  ; dvc repro                                         # DVC-orchestrated run
run:       ; python -m src.flows run                           # Prefect run
drift:     ; python -m src.flows drift
serve:     ; python -m src.flows serve                         # weekly train + daily drift check
ui:        ; mlflow ui --backend-store-uri sqlite:///mlflow.db
up:        ; docker compose up --build
