import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# Define the database path
DB_FILE = os.path.join(os.path.dirname(__file__), 'sqlite.db')
DB_URL = f'sqlite:///{DB_FILE}'

# Initialize database components
engine = create_engine(DB_URL, echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class User(Base):
    """Database model for a user."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    # The crucial column for user access control:
    role = Column(String(50), default='user', nullable=False) 

    # Relationship to prediction history
    history = relationship("NewsHistory", backref="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"

class NewsHistory(Base):
    """Database model for storing news prediction history."""
    __tablename__ = 'news_history'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    news_text = Column(String, nullable=False)
    prediction = Column(String(10), nullable=False) # 'Fake' or 'Real'
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<NewsHistory(id={self.id}, prediction='{self.prediction}', confidence={self.confidence})>"

# This line ensures tables are created when 'app.py' runs and calls create_all
if __name__ == '__main__':
    Base.metadata.create_all(engine)
