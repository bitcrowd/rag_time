from sqlalchemy import Column, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship
from .base import Base
from collections import Counter
from operator import itemgetter

class Path(Base):
    __tablename__ = 'paths'

    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String, nullable=False, unique=True)
    commit_paths = relationship("CommitPath", back_populates="path")

    __table_args__ = (UniqueConstraint('path', name='uix_path'),)

    def __str__(self):
        output = f"{self.path}\n\n"
        for commit_path in self.commit_paths:
            output += f"{commit_path.change_type}\t{commit_path.commit.hash}\t{commit_path.commit.short_message()}\n"
        
        # Count occurrences of related paths
        related_paths = Counter()
        for commit_path in self.commit_paths:
            for related_commit_path in commit_path.commit.commit_paths:
                related_path = related_commit_path.path
                if related_path != self:
                    related_paths[related_path.path] += 1
        
        # Filter out items with count less than 5
        filtered_related_paths = {path: count for path, count in related_paths.items() if count >= 5}
        
        # Sort related paths by count in descending order
        sorted_related_paths = sorted(
            filtered_related_paths.items(),
            key=itemgetter(1),
            reverse=True
        )

        # Apply the new condition
        if len(sorted_related_paths) > 10:
            max_count = sorted_related_paths[0][1]
            min_count = sorted_related_paths[-1][1]
            if max_count - min_count > 1:
                sorted_related_paths = sorted_related_paths[:10]
            else:
                sorted_related_paths = []

        output += f"\nRelated paths (path: count):\n"
        for path, count in sorted_related_paths:
            output += f"\t{path}: {count}\n"
        return output
