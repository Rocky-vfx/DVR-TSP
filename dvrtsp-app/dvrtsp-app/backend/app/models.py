from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    t_max = Column(Float, default=55.0)
    speed = Column(Float, default=42.0)
    hub_x = Column(Float, nullable=False)
    hub_y = Column(Float, nullable=False)
    nodes_json = Column(Text, nullable=False)   # JSON-encoded list of node dicts
    edges_json = Column(Text, nullable=False)   # JSON-encoded list of edge dicts
    source = Column(String, default="synthetic", nullable=True)
    meta_json = Column(Text, nullable=True)

    runs = relationship("Run", back_populates="scenario", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, index=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    algorithm = Column(String, nullable=False)
    total_value = Column(Float, nullable=False)
    total_time = Column(Float, nullable=False)
    count = Column(Integer, nullable=False)
    order_json = Column(Text, nullable=False)   # JSON-encoded list of node ids visited
    steps_json = Column(Text, nullable=False)   # JSON-encoded per-step timeline
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # New recovery metric columns (nullable for backward compat with existing rows)
    total_data_available = Column(Float, nullable=True)
    data_recovered = Column(Float, nullable=True)
    data_lost = Column(Float, nullable=True)
    lost_to_decay = Column(Float, nullable=True)
    lost_unvisited = Column(Float, nullable=True)
    recovery_percentage = Column(Float, nullable=True)
    time_budget = Column(Float, nullable=True)
    time_remaining = Column(Float, nullable=True)
    time_utilization_percentage = Column(Float, nullable=True)
    travel_time = Column(Float, nullable=True)
    recovery_time_total = Column(Float, nullable=True)
    nodes_visited = Column(Integer, nullable=True)
    nodes_total = Column(Integer, nullable=True)
    total_value_available = Column(Float, nullable=True)
    value_recovered = Column(Float, nullable=True)
    per_node_timeline_json = Column(Text, nullable=True)  # JSON per-node details
    summary_text = Column(Text, nullable=True)             # plain-English summary

    scenario = relationship("Scenario", back_populates="runs")
