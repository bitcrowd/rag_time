from sqlalchemy.orm import declarative_base
from sqlalchemy import inspect

BaseModel = declarative_base()

class Base(BaseModel):
    """Base model class that provides common functionality for all models"""
    __abstract__ = True  # This ensures SQLAlchemy won't try to create a table for this model

    def to_dict(self):
        """Convert model instance to dictionary"""
        return {c.key: getattr(self, c.key)
                for c in inspect(self).mapper.column_attrs}

    def __repr__(self):
        """Default string representation"""
        attrs = ', '.join(f'{k}={v}' for k, v in self.to_dict().items())
        return f"{self.__class__.__name__}({attrs})"

    @staticmethod
    def git_status_mappings(status_string: str) -> str:
        """Convert git status code to human readable string"""
        status_map = {
            ' ': 'file unmodified',
            'M': 'file modified',
            'T': 'file type changed',
            'A': 'file added',
            'D': 'file deleted',
            'R': 'file renamed',
            'C': 'file copied',
            'U': 'file updated but unmerged'
        }
        return status_map.get(status_string, 'unknown file status')
