
prod-rag/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── documents.py
│   │       └── ingestion_jobs.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── parsers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── pdf.py
│   │   │   └── registry.py
│   │   ├── chunking/
│   │   │   ├── __init__.py
│   │   │   ├── normalize.py
│   │   │   └── splitter.py
│   │   ├── embeddings/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── service.py
│   │   ├── indexing/
│   │   │   ├── __init__.py
│   │   │   └── service.py
│   │   └── service.py
│   │
│   ├── db/
│   │   ├── session.py
│   │   └── models/
│   │
│   ├── storage/
│   │   └── ...
│   │
│   └── workers/
│       └── ...
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
