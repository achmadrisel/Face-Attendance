from sqlalchemy import Column, Integer, String
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
from app.core.database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=False, nullable=False)
    label = Column(String, nullable=False)
    embedding = Column(Vector(512), nullable=False)
    path = Column(String, nullable=True)