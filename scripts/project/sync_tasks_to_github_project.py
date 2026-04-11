#!/usr/bin/env python3
"""
Sync tasks from docs/tasks.yaml to GitHub Project.

Usage:
    python scripts/project/sync_tasks_to_github_project.py \
        --project-number 123 \
        --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_tasks_config(config_path: Path = None) -> Dict[str, Any]:
    """Load tasks configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "docs" / "tasks.yaml"
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_gh_command(cmd: List[str], dry_run: bool = False) -> Optional[str]:
    """Run a gh CLI command and return output."""
    full_cmd = ["gh"] + cmd
    print(f"[{'DRY-RUN' if dry_run else 'EXEC'}] {' '.join(full_cmd)}")
    
    if dry_run:
        return None
    
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}", file=sys.stderr)
        return None


def get_issue_number_by_title(repo: str, title: str) -> Optional[str]:
    """Find issue by title."""
    cmd = [
        "issue", "list", 
        "--repo", repo,
        "--search", f'"{title}"',
        "--json", "number",
        "--limit", "1"
    ]
    output = run_gh_command(cmd, dry_run=False)
    
    if output:
        try:
            issues = json.loads(output)
            if issues:
                return str(issues[0]["number"])
        except (json.JSONDecodeError, IndexError):
            pass
    
    return None


def create_or_find_issue(
    repo: str,
    task_id: str,
    title: str,
    description: str,
    labels: List[str],
    milestone: Optional[str] = None,
    dry_run: bool = False
) -> Optional[str]:
    """Create issue or return existing one."""
    full_title = f"[{task_id}] {title}"
    
    # Try to find existing
    existing = get_issue_number_by_title(repo, task_id)
    if existing:
        print(f"  → Issue #{existing} already exists")
        return existing
    
    # Create new
    cmd = [
        "issue", "create",
        "--repo", repo,
        "--title", full_title,
        "--body", description,
    ]
    
    if labels:
        cmd.extend(["--label", ",".join(labels)])
    
    if milestone:
        cmd.extend(["--milestone", milestone])
    
    output = run_gh_command(cmd, dry_run=dry_run)
    
    if output and not dry_run:
        # Extract issue number from URL
        # Output is typically: https://github.com/owner/repo/issues/123
        issue_num = output.split("/")[-1]
        print(f"  ✓ Created issue #{issue_num}")
        return issue_num
    
    return None


def add_issue_to_project(
    repo: str,
    project_number: int,
    issue_number: str,
    dry_run: bool = False
) -> bool:
    """Add issue to project."""
    cmd = [
        "project", "item-add",
        str(project_number),
        "--owner", repo.split("/")[0],
        "--repo", repo.split("/")[1],
        "--issue-number", issue_number
    ]
    
    output = run_gh_command(cmd, dry_run=dry_run)
    return output is not None or dry_run


def add_labels_to_issue(
    repo: str,
    issue_number: str,
    labels: List[str],
    dry_run: bool = False
) -> bool:
    """Add labels to issue."""
    if not labels:
        return True
    
    cmd = [
        "issue", "edit",
        issue_number,
        "--repo", repo,
        "--add-label", ",".join(labels)
    ]
    
    output = run_gh_command(cmd, dry_run=dry_run)
    return output is not None or dry_run


def sync_tasks_to_project(
    project_number: int,
    repo: str = None,
    dry_run: bool = False,
    config_path: Path = None
) -> None:
    """Sync all tasks from YAML to GitHub Project."""
    
    config = load_tasks_config(config_path)
    
    if not repo:
        repo = f"{config['project_config']['owner']}/{config['project_config']['repo']}"
    
    if project_number is None:
        print("Error: --project-number is required or set project_number in tasks.yaml")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Syncing tasks to GitHub Project {project_number}")
    print(f"Repository: {repo}")
    print(f"{'='*60}\n")
    
    sprints = config.get("sprints", {})
    total_tasks = 0
    created_tasks = 0
    
    for sprint_id, sprint_data in sprints.items():
        print(f"\n📋 {sprint_data['title']}")
        print(f"   {sprint_data['start_date']} → {sprint_data['end_date']}")
        print()
        
        tasks = sprint_data.get("tasks", {})
        for task_id, task_data in tasks.items():
            total_tasks += 1
            
            print(f"  {task_id}: {task_data['title']}")
            
            issue_num = create_or_find_issue(
                repo=repo,
                task_id=task_id,
                title=task_data["title"],
                description=task_data.get("description", ""),
                labels=task_data.get("labels", []),
                milestone=sprint_data.get("milestone"),
                dry_run=dry_run
            )
            
            if issue_num:
                created_tasks += 1
                add_issue_to_project(
                    repo=repo,
                    project_number=project_number,
                    issue_number=issue_num,
                    dry_run=dry_run
                )
                add_labels_to_issue(
                    repo=repo,
                    issue_number=issue_num,
                    labels=task_data.get("labels", []),
                    dry_run=dry_run
                )
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Created/Found: {created_tasks}")
    if dry_run:
        print(f"  Mode: DRY-RUN (no changes made)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sync tasks from docs/tasks.yaml to GitHub Project"
    )
    parser.add_argument(
        "--project-number",
        type=int,
        required=False,
        help="GitHub Project number"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=False,
        help="GitHub repository (owner/repo)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        default=Path(__file__).parent.parent.parent / "docs" / "tasks.yaml",
        help="Path to tasks.yaml"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    sync_tasks_to_project(
        project_number=args.project_number,
        repo=args.repo,
        dry_run=args.dry_run,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
