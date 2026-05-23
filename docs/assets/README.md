# Demo And Screenshot Assets

This directory stores lightweight visual assets for the GitHub README, release page, and portfolio presentation.

## Expected Assets

| File | Status | How to create |
| :--- | :--- | :--- |
| `streamlit_ui.png` | placeholder expected | Run `python -m streamlit run streamlit_app.py --server.port 8502`, open the dashboard, and capture the main UI. |
| `api_docs.png` | placeholder expected | Run `uvicorn src.api.app:app --host 127.0.0.1 --port 8010`, open `/docs`, and capture the OpenAPI page. |
| `confusion_matrix.png` | generated | Copied from the current evaluation report when available. |
| `evaluation_summary.png` | placeholder expected | Capture the key metrics table or create a compact visual from `reports/evaluation/metrics_summary.json`. |
| `architecture_diagram.png` | existing or placeholder | Use the Mermaid diagram in `docs/ARCHITECTURE.md` or an existing generated architecture image. |
| `demo.gif` | placeholder expected | Record a short upload-to-prediction Streamlit/API demo. |

Do not invent screenshots or fake hosted URLs. Capture real local or deployed runs only.

## Existing Visuals

The repository already contains several presentation-oriented images in this directory. They can be used for README or portfolio layout, but should be replaced with fresh screenshots before a public release if the UI changes.
