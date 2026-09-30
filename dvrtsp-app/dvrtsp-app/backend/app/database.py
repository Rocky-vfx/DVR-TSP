from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./dvrtsp.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_migrations():
    """Non-destructive migration for existing SQLite databases:
    adds new columns to 'scenarios' and 'runs' tables if missing."""
    with engine.connect() as conn:
        # --- Scenarios table ---
        result = conn.exec_driver_sql("PRAGMA table_info(scenarios)").fetchall()
        cols = [r[1] for r in result]
        if cols:
            if "source" not in cols:
                conn.exec_driver_sql("ALTER TABLE scenarios ADD COLUMN source VARCHAR DEFAULT 'synthetic'")
            if "meta_json" not in cols:
                conn.exec_driver_sql("ALTER TABLE scenarios ADD COLUMN meta_json TEXT")

        # --- Runs table: new recovery metric columns ---
        result = conn.exec_driver_sql("PRAGMA table_info(runs)").fetchall()
        run_cols = [r[1] for r in result]
        if run_cols:
            new_float_cols = [
                "total_data_available", "data_recovered", "data_lost",
                "lost_to_decay", "lost_unvisited", "recovery_percentage",
                "time_budget", "time_remaining", "time_utilization_percentage",
                "travel_time", "recovery_time_total",
                "total_value_available", "value_recovered",
            ]
            new_int_cols = ["nodes_visited", "nodes_total"]
            new_text_cols = ["per_node_timeline_json", "summary_text"]

            for col in new_float_cols:
                if col not in run_cols:
                    conn.exec_driver_sql(f"ALTER TABLE runs ADD COLUMN {col} REAL")
            for col in new_int_cols:
                if col not in run_cols:
                    conn.exec_driver_sql(f"ALTER TABLE runs ADD COLUMN {col} INTEGER")
            for col in new_text_cols:
                if col not in run_cols:
                    conn.exec_driver_sql(f"ALTER TABLE runs ADD COLUMN {col} TEXT")

        conn.commit()

