# lab7_api.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import psycopg2
import time
import threading

from lab7_core import (
    connect_admin, connect_lab, setup_database, create_schema,
    generate_orders_data, insert_orders, benchmark_query,
    create_indexes, analyze_tables, get_table_sizes
)

app = FastAPI(
    title="PostgreSQL Partitioning Lab API",
    description="API для проведения лабораторной работы по партиционированию в PostgreSQL",
    version="1.0.0"
)

# Глобальное состояние (в реальном проекте — Redis или БД)
state = {
    "status": "idle",  # idle | running | completed | error
    "progress": 0,
    "results": {},
    "error": None,
    "start_time": None
}

class RunRequest(BaseModel):
    drop_and_recreate: Optional[bool] = True
    generate_new_data: Optional[bool] = True

@app.get("/")
async def root():
    return {"message": "PostgreSQL Partitioning Lab API", "status": state["status"]}

@app.get("/status")
async def get_status():
    return {
        "status": state["status"],
        "progress": state["progress"],
        "elapsed_sec": round(time.time() - state["start_time"], 2) if state["start_time"] else 0,
        "error": state["error"]
    }

@app.get("/results")
async def get_results():
    if state["status"] != "completed":
        raise HTTPException(status_code=400, detail="Lab not completed yet")
    return state["results"]

@app.post("/run")
async def run_lab(req: RunRequest = RunRequest(), background_tasks: BackgroundTasks = None):
    if state["status"] == "running":
        raise HTTPException(status_code=400, detail="Lab is already running")

    state.update({
        "status": "running",
        "progress": 0,
        "error": None,
        "start_time": time.time(),
        "results": {}
    })

    background_tasks.add_task(_run_lab_in_background, req)
    return {"message": "Lab started in background", "status": "running"}

def _run_lab_in_background(req: RunRequest):
    try:
        step = 0
        total_steps = 7

        def update_step(msg: str, inc: int = 1):
            nonlocal step
            step += inc
            state["progress"] = round(step / total_steps * 100, 1)
            print(f"[API] {msg} ({state['progress']}%)")

        # [1] Setup DB
        update_step("Connecting and setting up database", 0)
        admin_conn = connect_admin()
        if req.drop_and_recreate:
            setup_database(admin_conn)
        admin_conn.close()

        conn = connect_lab()

        # [2] Schema
        update_step("Creating schema")
        create_schema(conn)

        # [3] Data
        update_step("Generating and inserting orders")
        orders = generate_orders_data() if req.generate_new_data else []
        insert_result = insert_orders(conn, orders)

        # [4] Benchmark — before indexes
        update_step("Benchmarking (before indexes)")
        bench_before = {
            "regular": benchmark_query(conn, "orders", "2023-04-01", "2023-06-30"),
            "partitioned": benchmark_query(conn, "orders_partitioned", "2023-04-01", "2023-06-30")
        }

        # [5] Indexes
        update_step("Creating indexes")
        create_indexes(conn)

        # [6] Benchmark — after indexes
        update_step("Benchmarking (after indexes)")
        bench_after = {
            "regular": benchmark_query(conn, "orders", "2023-04-01", "2023-06-30"),
            "partitioned": benchmark_query(conn, "orders_partitioned", "2023-04-01", "2023-06-30")
        }

        # [7] Maintenance & Stats
        update_step("Running ANALYZE and collecting stats")
        analyze_time = analyze_tables(conn)
        sizes = get_table_sizes(conn)

        conn.close()

        # Save results
        state["results"] = {
            "insert": insert_result,
            "benchmark_before_indexes": bench_before,
            "benchmark_after_indexes": bench_after,
            "analyze_time_sec": analyze_time,
            "table_sizes": sizes,
            "total_time_sec": round(time.time() - state["start_time"], 2)
        }
        state["status"] = "completed"

    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        print(f"[ERROR] {e}")

# --- Utility Endpoints ---

@app.post("/reset")
async def reset_state():
    global state
    state = {
        "status": "idle",
        "progress": 0,
        "results": {},
        "error": None,
        "start_time": None
    }
    return {"message": "State reset"}

@app.get("/health")
async def health_check():
    try:
        conn = connect_admin()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        conn.close()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": str(e)}