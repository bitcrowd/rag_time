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

    def _format_commit_history(self):
        output = []
        for commit_path in self.commit_paths:
            output.append(f"{commit_path.change_type}\t{commit_path.commit.hash}\t{commit_path.commit.short_message()}")
        return "\n".join(output)

    def _get_related_paths_counts(self):
        related_paths = Counter()
        for commit_path in self.commit_paths:
            for related_commit_path in commit_path.commit.commit_paths:
                related_path = related_commit_path.path
                if related_path != self:
                    related_paths[related_path.path] += 1
        return related_paths

    def _filter_and_sort_related_paths(self, related_paths, min_count=5):
        filtered = {path: count for path, count in related_paths.items() if count >= min_count}
        return sorted(filtered.items(), key=itemgetter(1), reverse=True)

    def _apply_top_ten_filter(self, sorted_paths):
        if len(sorted_paths) <= 10:
            return sorted_paths
            
        max_count = sorted_paths[0][1]
        min_count = sorted_paths[-1][1]
        return sorted_paths[:10] if max_count - min_count > 1 else []

    def _format_related_paths(self, related_paths):
        return "\n".join(f"\t{path}: {count}" for path, count in related_paths)

    def __str__(self):
        parts = [
            f"{self.path}\n",
            self._format_commit_history(),
            "\nRelated paths (path: count):",
            self._format_related_paths(
                self._apply_top_ten_filter(
                    self._filter_and_sort_related_paths(
                        self._get_related_paths_counts()
                    )
                )
            )
        ]
        return "\n".join(parts)
