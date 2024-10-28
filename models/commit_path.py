from sqlalchemy import Column, String, ForeignKey, Integer
from sqlalchemy.orm import relationship
from .base import Base

class CommitPath(Base):
    __tablename__ = 'commit_paths'

    commit_hash = Column(String, ForeignKey('commits.hash'), primary_key=True)
    path_id = Column(Integer, ForeignKey('paths.id'), primary_key=True)
    change_type = Column(String)

    commit = relationship("Commit", back_populates="commit_paths")
    path = relationship("Path", back_populates="commit_paths")

    def __str__(self):
        return f"{self.change_type}\t{self.path.path}"
    
    def __repr__(self):
        return f"{self.change_type}\t{self.path.path}"
