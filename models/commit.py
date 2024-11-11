from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.orm import relationship
import textwrap
from .base import Base

class Commit(Base):
    __tablename__ = 'commits'

    hash = Column(String, primary_key=True)
    previous_commit = Column(String, ForeignKey('commits.hash'))
    next_commit = Column(String, ForeignKey('commits.hash'))
    message = Column(String)
    commit_paths = relationship("CommitPath", back_populates="commit")

    def __str__(self):
        file_changes = ""
        for commit_path in self.commit_paths:
            file_changes += f"{commit_path}\n"
        output = f"""
Commit {self.hash}
----- Begin Commit message ----
{self.message}
----- End Commit message ----

----- Begin File Changes ----
{file_changes}
----- End File Changes -----
            """
        return textwrap.dedent(output).strip()

    def short_message(self):
        """Return only the first line of the commit message."""
        return self.message.split('\n')[0] if self.message else ''
