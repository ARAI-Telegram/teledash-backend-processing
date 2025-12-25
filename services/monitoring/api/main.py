"""
Monitoring API for ARAI ML Processing Services

Provides real-time status indicators showing which ML analysis tasks
are currently running or queued across all processing services.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from celery import Celery
from pydantic import BaseModel

from common.settings import settings

app = FastAPI(title="ARAI Processing Monitor", version="0.1")

# Add CORS middleware to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Celery app for inspecting workers
celery_app = Celery(broker=f"redis://redis-processing:{settings.redis_port}/0")


class QueueStatus(BaseModel):
    """Status information for a single processing queue."""
    queue_name: str
    active_tasks: int
    queued_tasks: int
    worker_online: bool
    current_tasks: List[Dict[str, Any]]


class ProcessingStatus(BaseModel):
    """Overall processing status across all ML services."""
    timestamp: str
    queues: Dict[str, QueueStatus]
    total_active: int
    total_queued: int


class TaskInfo(BaseModel):
    """Detailed information about a running task."""
    task_id: str
    task_name: str
    worker: str
    args: Optional[str] = None
    started_at: Optional[float] = None


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "ARAI Processing Monitor API",
        "version": "0.1",
        "endpoints": {
            "/status": "Get overall ML processing status",
            "/status/{queue}": "Get status for specific queue",
            "/active": "Get all active tasks",
            "/workers": "Get worker information"
        }
    }


@app.get("/status", response_model=ProcessingStatus)
async def get_processing_status() -> ProcessingStatus:
    """
    Get real-time status of all ML processing queues.

    Returns status indicators showing which analyses are running/queued:
    - classification
    - sentiment
    - ner (named entity recognition)
    - topic_modeling
    - asr (automatic speech recognition)
    - text-embeddings
    - image-embeddings
    - ngrams
    - export
    - network_analysis
    - temporal_analysis
    """
    try:
        # Get Celery inspector
        inspect = celery_app.control.inspect()

        # Get active tasks per worker
        active_tasks_by_worker = inspect.active() or {}

        # Get reserved (queued) tasks per worker
        reserved_tasks_by_worker = inspect.reserved() or {}

        # Get registered tasks to identify which workers handle which queues
        registered = inspect.registered() or {}

        # Define all ML processing queues
        queue_names = [
            "classification",
            "sentiment",
            "ner",
            "topic_modeling",
            "asr",
            "text-embeddings",
            "image-embeddings",
            "ngrams",
            "export",
            "network_analysis",
            "temporal_analysis"
        ]

        queues_status = {}
        total_active = 0
        total_queued = 0

        for queue_name in queue_names:
            # Find workers for this queue (workers are named like "classification-worker@hostname")
            queue_workers = [w for w in registered.keys() if queue_name in w]

            active_count = 0
            queued_count = 0
            current_tasks = []
            worker_online = len(queue_workers) > 0

            # Count active and queued tasks for this queue's workers
            for worker in queue_workers:
                # Active tasks
                if worker in active_tasks_by_worker:
                    worker_active = active_tasks_by_worker[worker]
                    active_count += len(worker_active)

                    # Extract task details
                    for task in worker_active:
                        current_tasks.append({
                            "task_id": task.get("id", "unknown"),
                            "task_name": task.get("name", "unknown"),
                            "worker": worker,
                            "started_at": task.get("time_start", None)
                        })

                # Reserved/queued tasks
                if worker in reserved_tasks_by_worker:
                    queued_count += len(reserved_tasks_by_worker[worker])

            queues_status[queue_name] = QueueStatus(
                queue_name=queue_name,
                active_tasks=active_count,
                queued_tasks=queued_count,
                worker_online=worker_online,
                current_tasks=current_tasks
            )

            total_active += active_count
            total_queued += queued_count

        return ProcessingStatus(
            timestamp=datetime.utcnow().isoformat(),
            queues=queues_status,
            total_active=total_active,
            total_queued=total_queued
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get processing status: {str(e)}"
        )


@app.get("/status/{queue_name}", response_model=QueueStatus)
async def get_queue_status(queue_name: str) -> QueueStatus:
    """
    Get detailed status for a specific processing queue.

    Args:
        queue_name: Name of the queue (e.g., 'sentiment', 'ner', 'classification')
    """
    try:
        inspect = celery_app.control.inspect()

        active_tasks_by_worker = inspect.active() or {}
        reserved_tasks_by_worker = inspect.reserved() or {}
        registered = inspect.registered() or {}

        # Find workers for this queue
        queue_workers = [w for w in registered.keys() if queue_name in w]

        if not queue_workers:
            return QueueStatus(
                queue_name=queue_name,
                active_tasks=0,
                queued_tasks=0,
                worker_online=False,
                current_tasks=[]
            )

        active_count = 0
        queued_count = 0
        current_tasks = []

        for worker in queue_workers:
            if worker in active_tasks_by_worker:
                worker_active = active_tasks_by_worker[worker]
                active_count += len(worker_active)

                for task in worker_active:
                    current_tasks.append({
                        "task_id": task.get("id", "unknown"),
                        "task_name": task.get("name", "unknown"),
                        "worker": worker,
                        "started_at": task.get("time_start", None),
                        "args": str(task.get("args", []))[:100]  # Truncate args
                    })

            if worker in reserved_tasks_by_worker:
                queued_count += len(reserved_tasks_by_worker[worker])

        return QueueStatus(
            queue_name=queue_name,
            active_tasks=active_count,
            queued_tasks=queued_count,
            worker_online=True,
            current_tasks=current_tasks
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get queue status: {str(e)}"
        )


@app.get("/active")
async def get_active_tasks() -> Dict[str, Any]:
    """
    Get all currently active tasks across all workers.
    """
    try:
        inspect = celery_app.control.inspect()
        active = inspect.active() or {}

        all_tasks = []
        for worker, tasks in active.items():
            for task in tasks:
                all_tasks.append({
                    "task_id": task.get("id"),
                    "task_name": task.get("name"),
                    "worker": worker,
                    "started_at": task.get("time_start"),
                    "args": task.get("args", []),
                    "kwargs": task.get("kwargs", {})
                })

        return {
            "active_count": len(all_tasks),
            "tasks": all_tasks
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get active tasks: {str(e)}"
        )


@app.get("/workers")
async def get_workers_info() -> Dict[str, Any]:
    """
    Get information about all connected workers.
    """
    try:
        inspect = celery_app.control.inspect()

        stats = inspect.stats() or {}
        registered = inspect.registered() or {}
        active_queues = inspect.active_queues() or {}

        workers = []
        for worker_name in stats.keys():
            worker_info = {
                "name": worker_name,
                "status": "online",
                "registered_tasks": len(registered.get(worker_name, [])),
                "queues": [q["name"] for q in active_queues.get(worker_name, [])],
                "stats": stats.get(worker_name, {})
            }
            workers.append(worker_info)

        return {
            "worker_count": len(workers),
            "workers": workers
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get worker info: {str(e)}"
        )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add response time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    try:
        # Try to connect to Celery
        inspect = celery_app.control.inspect()
        inspect.ping()

        return {
            "status": "healthy",
            "service": "monitoring",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,  # Different port from text/image APIs
        reload=True
    )
