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
    
    # Create new (without labels/milestone to avoid failing when they don't exist yet)
    cmd = [
        "issue", "create",
        "--repo", repo,
        "--title", full_title,
        "--body", description,
    ]
    
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
    owner, repo_name = repo.split("/")
    issue_url = f"https://github.com/{owner}/{repo_name}/issues/{issue_number}"
    cmd = [
        "project", "item-add",
        str(project_number),
        "--owner", owner,
        "--url", issue_url
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


def ensure_label_exists(repo: str, label: str, dry_run: bool = False) -> bool:
    """Ensure a label exists in the repository."""
    if not label:
        return True

    check_cmd = [
        "label", "list",
        "--repo", repo,
        "--search", label,
        "--json", "name",
        "--limit", "100"
    ]
    output = run_gh_command(check_cmd, dry_run=False)
    if output:
        try:
            labels = json.loads(output)
            if any(item.get("name") == label for item in labels):
                return True
        except json.JSONDecodeError:
            pass

    create_cmd = [
        "label", "create",
        label,
        "--repo", repo,
        "--description", "Auto-created by task sync script"
    ]
    return run_gh_command(create_cmd, dry_run=dry_run) is not None or dry_run


def ensure_milestone_exists(repo: str, milestone: str, dry_run: bool = False) -> bool:
    """Ensure a milestone exists in the repository."""
    if not milestone:
        return True

    list_cmd = [
        "api",
        f"repos/{repo}/milestones?state=all&per_page=100"
    ]
    output = run_gh_command(list_cmd, dry_run=False)
    if output:
        try:
            milestones = json.loads(output)
            if any(item.get("title") == milestone for item in milestones):
                return True
        except json.JSONDecodeError:
            pass

    create_cmd = [
        "api", "-X", "POST",
        f"repos/{repo}/milestones",
        "-f", f"title={milestone}"
    ]
    return run_gh_command(create_cmd, dry_run=dry_run) is not None or dry_run


def set_issue_milestone(
    repo: str,
    issue_number: str,
    milestone: Optional[str],
    dry_run: bool = False
) -> bool:
    """Set milestone on an issue."""
    if not milestone:
        return True

    cmd = [
        "issue", "edit",
        issue_number,
        "--repo", repo,
        "--milestone", milestone
    ]
    return run_gh_command(cmd, dry_run=dry_run) is not None or dry_run


def get_project_id(owner: str, project_number: int) -> Optional[str]:
    """Get GitHub Project node ID from owner and project number."""
    cmd = [
        "project", "list",
        "--owner", owner,
        "--format", "json"
    ]
    output = run_gh_command(cmd, dry_run=False)
    if not output:
        return None

    try:
        data = json.loads(output)
        for project in data.get("projects", []):
            if int(project.get("number", -1)) == int(project_number):
                return project.get("id")
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

    return None


def ensure_project_date_field(owner: str, project_number: int, field_name: str, dry_run: bool = False) -> Optional[str]:
    """Ensure a DATE field exists in the project and return its field ID."""
    list_cmd = [
        "project", "field-list",
        str(project_number),
        "--owner", owner,
        "--format", "json"
    ]
    output = run_gh_command(list_cmd, dry_run=False)
    if output:
        try:
            fields = json.loads(output).get("fields", [])
            for field in fields:
                if field.get("name") == field_name:
                    return field.get("id")
        except json.JSONDecodeError:
            pass

    create_cmd = [
        "project", "field-create",
        str(project_number),
        "--owner", owner,
        "--name", field_name,
        "--data-type", "DATE"
    ]
    run_gh_command(create_cmd, dry_run=dry_run)

    output = run_gh_command(list_cmd, dry_run=False)
    if output:
        try:
            fields = json.loads(output).get("fields", [])
            for field in fields:
                if field.get("name") == field_name:
                    return field.get("id")
        except json.JSONDecodeError:
            pass

    return None


def get_project_status_field(owner: str, project_number: int) -> tuple[Optional[str], Dict[str, str]]:
    """Return Status field id and normalized option-name -> option-id mapping."""
    list_cmd = [
        "project", "field-list",
        str(project_number),
        "--owner", owner,
        "--format", "json"
    ]
    output = run_gh_command(list_cmd, dry_run=False)
    if not output:
        return None, {}

    try:
        fields = json.loads(output).get("fields", [])
    except json.JSONDecodeError:
        return None, {}

    for field in fields:
        if field.get("name") == "Status" and field.get("type") == "ProjectV2SingleSelectField":
            mapping: Dict[str, str] = {}
            for option in field.get("options", []):
                name = str(option.get("name", "")).strip().lower()
                option_id = option.get("id")
                if name and option_id:
                    mapping[name] = option_id
            return field.get("id"), mapping

    return None, {}


def normalize_task_status(task_status: Optional[str]) -> str:
    """Normalize YAML task status names to Project status option names."""
    raw = str(task_status or "").strip().lower()
    if raw in {"todo", "to-do", "to do", "not-started", "not started", "pending"}:
        return "todo"
    if raw in {"in-progress", "in progress", "doing", "wip", "ongoing"}:
        return "in progress"
    if raw in {"done", "completed", "complete", "finished"}:
        return "done"
    return "todo"


def set_project_item_status(
    item_id: str,
    project_id: str,
    status_field_id: str,
    status_option_id: str,
    dry_run: bool = False
) -> bool:
    """Set single-select Status field on a project item."""
    if not (item_id and project_id and status_field_id and status_option_id):
        return True

    cmd = [
        "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", status_field_id,
        "--single-select-option-id", status_option_id
    ]
    return run_gh_command(cmd, dry_run=dry_run) is not None or dry_run


def set_project_item_date(
    item_id: str,
    project_id: str,
    field_id: str,
    value: Optional[str],
    dry_run: bool = False
) -> bool:
    """Set DATE field value on a project item."""
    if not (item_id and project_id and field_id and value):
        return True

    cmd = [
        "project", "item-edit",
        "--id", item_id,
        "--project-id", project_id,
        "--field-id", field_id,
        "--date", value
    ]
    return run_gh_command(cmd, dry_run=dry_run) is not None or dry_run


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

    owner = repo.split("/")[0]
    project_id = get_project_id(owner=owner, project_number=project_number)
    start_date_field_id = ensure_project_date_field(owner=owner, project_number=project_number, field_name="Start date", dry_run=dry_run)
    target_date_field_id = ensure_project_date_field(owner=owner, project_number=project_number, field_name="Target date", dry_run=dry_run)
    status_field_id, status_options = get_project_status_field(owner=owner, project_number=project_number)
    
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

        sprint_milestone = sprint_data.get("milestone")
        ensure_milestone_exists(repo=repo, milestone=sprint_milestone, dry_run=dry_run)
        
        tasks = sprint_data.get("tasks", {})
        for task_id, task_data in tasks.items():
            total_tasks += 1
            
            print(f"  {task_id}: {task_data['title']}")

            labels = task_data.get("labels", [])
            for label in labels:
                ensure_label_exists(repo=repo, label=label, dry_run=dry_run)
            
            issue_num = create_or_find_issue(
                repo=repo,
                task_id=task_id,
                title=task_data["title"],
                description=task_data.get("description", ""),
                labels=labels,
                milestone=sprint_milestone,
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

                item_list_cmd = [
                    "project", "item-list",
                    str(project_number),
                    "--owner", owner,
                    "--format", "json"
                ]
                item_list_output = run_gh_command(item_list_cmd, dry_run=False)
                item_id = None
                if item_list_output:
                    try:
                        items = json.loads(item_list_output).get("items", [])
                        for item in items:
                            content = item.get("content", {})
                            if str(content.get("number", "")) == str(issue_num):
                                item_id = item.get("id")
                                break
                    except json.JSONDecodeError:
                        item_id = None

                set_project_item_date(
                    item_id=item_id,
                    project_id=project_id,
                    field_id=start_date_field_id,
                    value=sprint_data.get("start_date"),
                    dry_run=dry_run
                )
                set_project_item_date(
                    item_id=item_id,
                    project_id=project_id,
                    field_id=target_date_field_id,
                    value=sprint_data.get("end_date"),
                    dry_run=dry_run
                )

                normalized_status = normalize_task_status(task_data.get("status"))
                status_option_id = status_options.get(normalized_status)
                set_project_item_status(
                    item_id=item_id,
                    project_id=project_id,
                    status_field_id=status_field_id,
                    status_option_id=status_option_id,
                    dry_run=dry_run
                )

                add_labels_to_issue(
                    repo=repo,
                    issue_number=issue_num,
                    labels=labels,
                    dry_run=dry_run
                )
                set_issue_milestone(
                    repo=repo,
                    issue_number=issue_num,
                    milestone=sprint_milestone,
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
