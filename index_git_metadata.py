import sys
import os
import argparse
import random
import re
import subprocess
import textwrap
from git import Repo
from sqlalchemy import create_engine, Column, String, ForeignKey, Integer, UniqueConstraint

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from models import Base, Commit, Path, CommitPath

def connect_or_create_database(db_path, clean=False):
    """Create the SQLite database and commits table."""
    if clean and os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing database: {db_path}")

    engine = create_engine(f'sqlite:///{db_path}')
    if not os.path.exists(db_path):
        Base.metadata.create_all(engine)

    
    return sessionmaker(bind=engine)()

def fetch_commit_message(repo_path, hash):
    os.chdir(repo_path)
    result = subprocess.run(
        ['git', 'log', '--format=%B', '-n', '1', hash],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()

def fetch_commit_list(repo):
    """Retrieve commit hashes from the main branch."""
    commits = []
    for commit in repo.iter_commits('main', reverse=True):
        commits.append(Commit(hash=commit.hexsha))
    return commits

def fetch_commit_data(repo, commits, session):
    for commit in commits:
        commit_obj = repo.commit(commit.hash)
        commit.message = commit_obj.message.strip()
        
        commit_paths = fetch_commit_paths(repo, commit, session)
        ## go through the list of paths and create path objects, add them to the session
        for _object_mode, path in commit_paths:
            path_obj = session.query(Path).filter_by(path=path).first()
            if not path_obj:
                path_obj = Path(path=path)
                session.add(path_obj)
            commit_path = CommitPath(
                commit=commit,
                path=path_obj,
                change_type=_object_mode
            )
            session.add(commit_path)
        
    return commits

def link_commit_ancestry(commits):
    for index, commit in enumerate(commits):
        commit.previous_commit = commits[index - 1].hash if index > 0 else None
        commit.next_commit = commits[index + 1].hash if index < len(commits) - 1 else None
    return commits

def store_commits(session, commits):
    """Store commit data in the database."""
    for commit in commits:
        session.merge(commit)
    session.commit()

def clone_or_update_repository(repo_url, repo_path):
    """Clone the repository if it doesn't exist, or update it if it does."""
    if not os.path.exists(repo_path):
        print(f"Cloning repository from {repo_url} to {repo_path}")
        repo = Repo.clone_from(repo_url, repo_path)
    else:
        print(f"Repository already exists at {repo_path}")
        print("Updating repository...")
        repo = Repo(repo_path)
        origin = repo.remotes.origin
        origin.fetch()
        origin.pull()
    return repo

def parse_changed_files(diff_output):
    """Parse the git diff output and return a list of (change_type, path) tuples."""
    changed_files = []
    for line in diff_output.strip().split('\n'):
        match = re.match(r'(.)\s+(\S.*)', line)
        if match:
            object_mode, path = match.groups()
            changed_files.append((object_mode, path))
    return changed_files

def fetch_commit_paths(repo, commit, session):
    """Process the paths changed in a commit and create corresponding database entries."""
    
    # Get the diff output
    diff_output = repo.git.diff_tree('--no-commit-id', '--name-status', '-r', commit.hash)
    
    # Parse the changed files
    changed_files = parse_changed_files(diff_output)
    return changed_files

def ingest_git_metadata(repo_url, repo_path, db_path, clean):
    repo = clone_or_update_repository(repo_url, repo_path)
    session = connect_or_create_database(db_path, clean)
    try:
        commits = fetch_commit_list(repo)
        commits = link_commit_ancestry(commits)
        commits = fetch_commit_data(repo, commits, session)
        store_commits(session, commits)
        print(f"Stored {len(commits)} commits in the database {db_path}.")
        # random_commit = random.choice(commits)
        # retrieved_commit = session.query(Commit).filter(Commit.hash == random_commit.hash).first()
        # print(retrieved_commit)
        # # for each path of the commit, print the path object
        # for commit_path in retrieved_commit.commit_paths:
        #     print(commit_path.path)
    finally:
        session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store Git metadata in a SQLite database.")
    parser.add_argument("-u", "--repo-url", required=True, help="URL of the Git repository")
    parser.add_argument("-w", "--work-dir", required=True, help="Path to store or find the Git repository")
    parser.add_argument("-md", "--metadata-db-dir", required=True, help="Directory to store the metadata SQLite database")
    parser.add_argument("-mn", "--metadata-db-name", default="git_metadata.db", help="Name of the SQLite database file (default: git_metadata.db)")
    parser.add_argument("-c", "--clean", action="store_true", help="Delete existing database and create a new one instead of using the existing one")
    
    args = parser.parse_args()

    repo_url = args.repo_url
    repo_path = os.path.abspath(args.work_dir)
    db_dir = os.path.abspath(args.metadata_db_dir)
    db_path = os.path.join(db_dir, args.metadata_db_name)

    if not os.path.isdir(db_dir):
        os.makedirs(db_dir)
        print(f"Created database directory: {db_dir}")

    ingest_git_metadata(repo_url, repo_path, db_path, args.clean)
