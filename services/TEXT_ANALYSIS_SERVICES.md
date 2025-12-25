# Text Analysis Services

This document describes the new text analysis services added to ARAI for comprehensive linguistic and semantic analysis of Telegram messages.

## Overview

The text analysis suite consists of 5 services:
1. **Sentiment Analysis** - Emotional tone detection (positive/neutral/negative)
2. **Named Entity Recognition (NER)** - Extract and link people, organizations, locations, events
3. **N-gram Analysis** - Discover common phrases and terminology patterns
4. **Export Service** - Generate reports in CSV, PDF, and interactive HTML
5. **Enhanced Topic Modeling** - Topic evolution tracking and hierarchical clustering

## Service Profiles

All services can be started using Docker Compose profiles:

```bash
# Start individual services
docker-compose --profile sentiment up -d
docker-compose --profile ner up -d
docker-compose --profile ngrams up -d
docker-compose --profile export up -d
docker-compose --profile topic-modeling up -d

# Start all text analysis services
docker-compose --profile text-analysis up -d

# Start everything (all processing services)
docker-compose --profile all up -d
```

## Service Details

### 1. Sentiment Analysis Service

**Profile:** `sentiment` or `text-analysis`
**Queue:** `sentiment`
**Memory:** 4GB
**CPUs:** 2
**GPU:** Optional (uncomment reservations in docker-compose.yml)

**Model:** `oliverguhr/german-sentiment-bert` (with multilingual fallback)

**Auto-processing:** Every 60 minutes (configurable in Celery beat schedule)

**Tasks:**
- `sentiment.init_sentiment_analysis` - Initialize processing for new messages
- `sentiment.analyze_batch` - Analyze batch of 265 messages
- `sentiment.compute_stats` - Compute aggregate statistics

**API Endpoints:**
- `GET /text-analysis/sentiment/stats` - Aggregate sentiment statistics
- `GET /text-analysis/sentiment/timeline` - Time-series sentiment data
- `GET /text-analysis/sentiment/by-chat` - Per-chat sentiment breakdown
- `POST /text-analysis/sentiment/trigger` - Manual trigger for specific corpus

**Dependencies:**
- transformers 4.44.2
- torch 2.0.0+
- Redis (task queue)
- Elasticsearch (data storage)

---

### 2. Named Entity Recognition (NER) Service

**Profile:** `ner` or `text-analysis`
**Queue:** `ner`
**Memory:** 4GB
**CPUs:** 2
**GPU:** No

**Model:** `de_core_news_lg` (spaCy German language model)

**Auto-processing:** Every 60 minutes

**Tasks:**
- `ner.init_ner_extraction` - Initialize NER extraction
- `ner.extract_entities_batch` - Extract entities from 32 messages
- `ner.build_entity_network` - Build co-occurrence network
- `ner.consolidate_entities` - Merge duplicate entities

**Entity Types:**
- PERSON - People, fictional characters
- ORG - Organizations, companies, institutions
- LOC - Locations, cities, countries
- EVENT - Named events, conferences
- MISC - Other named entities

**API Endpoints:**
- `GET /text-analysis/entities` - List all entities (paginated)
- `GET /text-analysis/entities/{entity_id}` - Entity details
- `GET /text-analysis/entities/{entity_id}/messages` - Messages mentioning entity
- `GET /text-analysis/entities/network` - Co-occurrence network data (D3.js format)
- `POST /text-analysis/entities/trigger` - Manual NER trigger

**Dependencies:**
- spacy 3.7.0
- de_core_news_lg (German model)
- Redis, Elasticsearch

---

### 3. N-gram Analysis Service

**Profile:** `ngrams` or `text-analysis`
**Queue:** `ngrams`
**Memory:** 2GB
**CPUs:** 1
**GPU:** No

**Processing:** Manual trigger only (user-defined corpus)

**Tasks:**
- `ngrams.generate_ngrams` - Generate bi-grams, tri-grams for corpus
- `ngrams.word_frequency` - Compute word frequency distribution

**Features:**
- TF-IDF scoring for n-grams
- Stopword filtering (200+ German, 50+ English)
- Configurable n-values (2, 3, 4, etc.)
- Minimum frequency thresholds

**API Endpoints:**
- `GET /text-analysis/ngrams` - Get n-grams for corpus
- `POST /text-analysis/ngrams/generate` - Generate n-grams with filters
- `GET /text-analysis/word-frequency` - Word frequency distribution

**Dependencies:**
- nltk 3.8.0
- scikit-learn 1.3.0
- Redis, Elasticsearch

---

### 4. Export Service

**Profile:** `export` or `text-analysis`
**Queue:** `export`
**Memory:** 3GB
**CPUs:** 2
**Concurrency:** 2 (can handle 2 exports simultaneously)

**Storage:** MinIO/S3 with 7-day presigned URLs

**Tasks:**
- `export.generate_csv` - Export analysis results as CSV
- `export.generate_pdf` - Generate PDF report with charts
- `export.generate_html` - Generate interactive HTML dashboard
- `export.cleanup_old_exports` - Delete exports older than 7 days (runs daily at 3:30 AM)

**Export Formats:**

1. **CSV Export**
   - Flattened nested structures (dot notation)
   - Streaming for large datasets
   - Field selection support

2. **PDF Report**
   - Professional styling with weasyprint
   - Static charts (matplotlib)
   - Summary statistics
   - Data tables (limited to 50 rows per analysis type)

3. **Interactive HTML**
   - Standalone dashboard (no external dependencies)
   - Embedded Plotly.js charts
   - Responsive design
   - Client-side interactivity

**API Endpoints:**
- `POST /text-analysis/export` - Trigger export (returns task_id)
- `GET /text-analysis/export/{task_id}` - Download export file

**Dependencies:**
- weasyprint 60.1 (PDF generation)
- matplotlib 3.8.0 (static charts)
- plotly 5.18.0 (interactive charts)
- jinja2 3.1.2 (templates)
- boto3 1.28.0 (S3/MinIO client)
- Redis, Elasticsearch

---

### 5. Enhanced Topic Modeling Service

**Profile:** `topic-modeling` or `text-analysis`
**Queue:** `topic_modeling`
**Memory:** 8GB
**CPUs:** 4
**GPU:** Optional

**Model:** BERTopic with German sentence-transformers

**Enhancements (New Features):**

1. **Topic Evolution Tracking**
   - Task: `topic_modeling.compute_topic_evolution`
   - Endpoint: `GET /topics/{topic_id}/evolution`
   - Tracks how topics change over time
   - Shows frequency and top words per time period
   - Granularity: day, week, month

2. **Topic Hierarchy**
   - Task: `topic_modeling.build_topic_hierarchy`
   - Endpoint: `GET /topics/hierarchy`
   - Hierarchical clustering of topics
   - Tree structure based on topic similarity
   - Uses cosine distance + Ward linkage

3. **Topic Merging**
   - Task: `topic_modeling.merge_topics`
   - Endpoint: `POST /topics/merge`
   - Merge similar topics into one
   - Updates all message assignments
   - Preserves merge history

**Existing Features:**
- Automatic topic discovery
- Message-topic assignments
- Extremism category detection
- Representative documents

---

## Resource Requirements

### Minimum System Requirements

- **CPU:** 8 cores recommended (can run on 4 cores with reduced performance)
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 50GB for models and temporary files
- **GPU:** Optional but recommended for sentiment analysis (speeds up by 5-10x)

### Per-Service Resource Allocation

| Service | Memory | CPUs | GPU | Batch Size |
|---------|--------|------|-----|------------|
| Sentiment | 4GB | 2 | Optional | 265 messages |
| NER | 4GB | 2 | No | 32 messages |
| N-grams | 2GB | 1 | No | Full corpus |
| Export | 3GB | 2 | No | N/A |
| Topic Modeling | 8GB | 4 | Optional | 10,000 messages |

### Scaling Recommendations

**High-Volume Setup (>1M messages/day):**
```bash
# Scale sentiment workers
docker-compose up -d --scale sentiment=3

# Scale NER workers
docker-compose up -d --scale ner=2

# Scale export workers
docker-compose up -d --scale export=3
```

**Low-Resource Setup (<=100K messages/day):**
- Run only CPU versions (comment out GPU reservations)
- Reduce concurrency in Celery commands
- Use single worker per service

---

## Configuration

### Environment Variables

Add to `.env` file in `teledash-backend-processing/`:

```bash
# GPU Configuration
GPU_USE=false  # Set to true for GPU-accelerated sentiment analysis
GPU_DEVICE=0   # GPU device ID (if multiple GPUs)

# Sentiment Analysis
SENTIMENT_MODEL=oliverguhr/german-sentiment-bert
SENTIMENT_BATCH_SIZE=265

# NER Configuration
NER_MODEL=de_core_news_lg
NER_BATCH_SIZE=32
NER_MIN_COOCCURRENCE=2  # Minimum co-occurrences for network

# N-gram Configuration
NGRAM_MIN_FREQUENCY=5
NGRAM_STOPWORDS_LANG=de  # de, en, or multilingual

# Export Configuration
MINIO_ENDPOINT=http://minio:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_BUCKET=arai-exports
EXPORT_EXPIRATION_DAYS=7

# Topic Modeling
TOPIC_MIN_CHAR_LENGTH=50
TOPIC_MIN_MESSAGES=100
```

### Celery Beat Schedule (Auto-processing)

Configured in `teledash-backend/worker/worker/config.py`:

```python
beat_schedule = {
    "auto-sentiment-analysis": {
        "task": "sentiment.init_sentiment_analysis",
        "schedule": timedelta(minutes=60),  # Run every hour
    },
    "auto-ner-extraction": {
        "task": "ner.init_ner_extraction",
        "schedule": timedelta(minutes=60),  # Run every hour
    },
    "cleanup-old-exports": {
        "task": "export.cleanup_old_exports",
        "schedule": crontab(hour=3, minute=30),  # Daily at 3:30 AM
    },
}
```

---

## Usage Examples

### 1. Triggering Sentiment Analysis

```bash
# Via API
curl -X POST "http://localhost:8000/text-analysis/sentiment/trigger" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "chat_ids": ["chat_123", "chat_456"],
    "date_from": "2024-01-01",
    "date_to": "2024-12-31"
  }'

# Response
{
  "task_id": "abc123...",
  "status": "accepted",
  "message": "Sentiment analysis task queued"
}
```

### 2. Getting Sentiment Statistics

```bash
curl -X GET "http://localhost:8000/text-analysis/sentiment/stats?chat_ids=chat_123&date_from=2024-01-01" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response
{
  "total_messages": 15234,
  "sentiment_breakdown": {
    "positive": 4567,
    "neutral": 8901,
    "negative": 1766
  },
  "average_score": 0.34,
  "by_chat": [...]
}
```

### 3. Extracting Entities

```bash
# Get all entities
curl -X GET "http://localhost:8000/text-analysis/entities?type=PERSON&min_frequency=10" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get entity network
curl -X GET "http://localhost:8000/text-analysis/entities/network?chat_ids=chat_123&min_cooccurrence=5" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response (D3.js format)
{
  "nodes": [
    {"id": "entity_001", "name": "Angela Merkel", "type": "PERSON", "count": 234},
    {"id": "entity_002", "name": "CDU", "type": "ORG", "count": 189}
  ],
  "edges": [
    {"source": "entity_001", "target": "entity_002", "weight": 87}
  ]
}
```

### 4. Generating N-grams

```bash
curl -X POST "http://localhost:8000/text-analysis/ngrams/generate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "chat_ids": ["chat_123"],
    "n_values": [2, 3],
    "min_frequency": 10,
    "stopwords_language": "de"
  }'

# Response
{
  "task_id": "xyz789...",
  "status": "accepted",
  "estimated_time": 120
}
```

### 5. Exporting Analysis Results

```bash
# Export to PDF
curl -X POST "http://localhost:8000/text-analysis/export" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "format": "pdf",
    "analysis_types": ["sentiment", "entities", "ngrams"],
    "filters": {
      "chat_ids": ["chat_123"],
      "date_from": "2024-01-01",
      "date_to": "2024-12-31"
    },
    "include_visualizations": true
  }'

# Response
{
  "task_id": "export_abc123",
  "status": "accepted",
  "message": "Export generation started"
}

# Download (after task completes)
curl -X GET "http://localhost:8000/text-analysis/export/export_abc123" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  --output analysis_report.pdf
```

### 6. Topic Evolution

```bash
# Compute evolution for a topic
curl -X GET "http://localhost:8000/topics/topic_0001/evolution?granularity=week" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response
{
  "topic_id": "topic_0001",
  "evolution": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "frequency": 234,
      "top_words": ["migration", "grenze", "abschiebung", "asyl", "flüchtlinge"]
    },
    {
      "timestamp": "2024-01-08T00:00:00Z",
      "frequency": 189,
      "top_words": ["migration", "abschiebung", "dublin", "asyl", "integration"]
    }
  ]
}
```

### 7. Topic Hierarchy

```bash
curl -X GET "http://localhost:8000/topics/hierarchy" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response
{
  "hierarchy_id": "topic_hierarchy_v1",
  "num_topics": 47,
  "tree": {
    "type": "branch",
    "distance": 12.34,
    "children": [
      {"id": "topic_0001", "type": "leaf", "distance": 0},
      {
        "type": "branch",
        "distance": 5.67,
        "children": [...]
      }
    ]
  }
}
```

### 8. Merging Topics

```bash
curl -X POST "http://localhost:8000/topics/merge" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "topic_ids": ["topic_0012", "topic_0034", "topic_0056"],
    "new_label": "Immigration Policy Discussion"
  }'

# Response
{
  "task_id": "merge_xyz789",
  "status": "accepted",
  "message": "Merging 3 topics. The merge operation is processing."
}
```

---

## Monitoring and Troubleshooting

### Check Service Status

```bash
# View running containers
docker-compose ps

# Check worker logs
docker-compose logs -f sentiment
docker-compose logs -f ner
docker-compose logs -f export

# Check Celery queue status
docker exec -it sentiment celery -A worker.sentiment.tasks inspect active
docker exec -it ner celery -A worker.ner.tasks inspect active
```

### Common Issues

**1. Out of Memory Errors**
```
Solution: Increase memory limits in docker-compose.yml or reduce batch sizes
```

**2. Model Download Failures**
```bash
# Sentiment model
docker exec -it sentiment python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='oliverguhr/german-sentiment-bert')"

# NER model
docker exec -it ner python -m spacy download de_core_news_lg
```

**3. Slow Processing**
```
- Enable GPU support for sentiment analysis
- Increase worker concurrency
- Scale up number of workers
```

**4. Export File Not Found**
```
- Check MinIO/S3 configuration
- Verify presigned URL expiration (7 days default)
- Check export service logs for errors
```

---

## Performance Benchmarks

Tested on: AMD Ryzen 9 5950X, 64GB RAM, NVIDIA RTX 3080

| Service | Messages/Hour (CPU) | Messages/Hour (GPU) | Memory Usage |
|---------|---------------------|---------------------|--------------|
| Sentiment | ~15,000 | ~75,000 | 3.2GB |
| NER | ~8,000 | N/A | 3.5GB |
| N-grams | ~50,000 | N/A | 1.8GB |
| Topic Modeling | ~6,000 | ~12,000 | 6.5GB |

*Export service performance depends on format and data size*

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     ARAI Backend API                        │
│  /text-analysis/* endpoints                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Redis Task Queue                         │
│  sentiment | ner | ngrams | export | topic_modeling         │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┴─────────────────┬──────────────┬──────────┐
        ▼                           ▼              ▼          ▼
┌───────────────┐         ┌──────────────┐  ┌──────────┐  ┌────────┐
│   Sentiment   │         │     NER      │  │ N-grams  │  │ Export │
│   Worker      │         │   Worker     │  │  Worker  │  │ Worker │
│               │         │              │  │          │  │        │
│ ├─ analyze    │         │ ├─ extract  │  │ ├─ gen   │  │ ├─ csv │
│ ├─ compute    │         │ ├─ network  │  │ └─ freq  │  │ ├─ pdf │
│ └─ trigger    │         │ └─ merge    │  │          │  │ └─ html│
└───────┬───────┘         └──────┬───────┘  └────┬─────┘  └───┬────┘
        │                        │               │            │
        └────────────────────────┴───────────────┴────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │    Elasticsearch        │
                    │ ├─ sentiment_results    │
                    │ ├─ entities             │
                    │ ├─ entity_mentions      │
                    │ ├─ ngrams               │
                    │ └─ messages_* (updated) │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   MinIO/S3 Storage      │
                    │   exports/              │
                    │   ├─ sentiment/*.pdf    │
                    │   ├─ entities/*.csv     │
                    │   └─ ngrams/*.html      │
                    └─────────────────────────┘
```

---

## Data Flow

### Sentiment Analysis Flow
```
1. User triggers analysis OR auto-processing (every 60 min)
2. API creates task → Redis queue
3. Sentiment worker:
   - Fetches unanalyzed messages (batch of 265)
   - Loads BERT model (lazy loading)
   - Runs inference
   - Stores results in sentiment_results index
   - Updates messages_* with sentiment field
4. Compute stats task aggregates results
5. Frontend fetches via /sentiment/stats endpoint
```

### Entity Network Flow
```
1. NER worker extracts entities from messages (batch of 32)
2. Creates entity_mentions (individual occurrences)
3. Consolidates into entities (unique, aggregated)
4. Network builder computes co-occurrences:
   - Groups mentions by message
   - Builds pairwise entity matrix
   - Filters by min_cooccurrence threshold
5. Stores co-occurrence data in entities documents
6. Frontend requests /entities/network
7. API returns D3.js-compatible JSON
```

---

## Security Considerations

1. **API Authentication**: All endpoints require Bearer token authentication
2. **Export Storage**: Presigned URLs expire after 7 days
3. **Resource Limits**: Memory and CPU limits prevent DoS
4. **Input Validation**: All user inputs validated via Pydantic models
5. **SQL Injection**: No SQL used (Elasticsearch DSL only)
6. **File Access**: Export service sandboxed to MinIO bucket

---

## Future Enhancements

Potential additions for future versions:

1. **Advanced Analytics**
   - Emotion detection (beyond sentiment)
   - Sarcasm/irony detection
   - Hate speech classification

2. **Performance**
   - Distributed processing (Celery on multiple machines)
   - Result caching with Redis
   - Incremental model updates

3. **Visualization**
   - Real-time sentiment dashboards
   - Interactive entity timelines
   - Topic flow diagrams (Sankey)

4. **Integration**
   - Webhook notifications on analysis completion
   - Scheduled exports
   - External storage backends (Google Drive, Dropbox)

---

## Support

For issues, questions, or contributions:
- GitHub Issues: [ARAI Repository]
- Documentation: `/docs/text-analysis`
- Model Cards: See individual service README files
