#!/usr/bin/env python3
"""
Reverse sync from GitHub Project to tasks.yaml and cronograma_fases_pesquisa.md
Reads the current state of GitHub Project and updates local files accordingly.

This is the reverse of sync_tasks_to_github_project.py:
- Gets all items from Project
- Updates tasks.yaml with new status
- Updates cronograma_fases_pesquisa.md with current phase status
"""

import subprocess
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def run_gh_command(cmd: List[str]) -> str:
    """Run a GitHub CLI command and return JSON output."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {' '.join(cmd)}")
        print(f"stderr: {result.stderr}")
        return ""
    return result.stdout


def get_project_items(owner: str, project_number: int) -> List[Dict]:
    """Fetch all items from GitHub Project."""
    cmd = [
        "gh",
        "project",
        "item-list",
        str(project_number),
        "--owner",
        owner,
        "--format",
        "json",
    ]
    output = run_gh_command(cmd)
    if not output:
        return []
    try:
        data = json.loads(output)
        return data.get("items", [])
    except json.JSONDecodeError:
        return []


def extract_task_id(title: str) -> Optional[str]:
    """Extract task ID from title like '[T-06-01] Title Here'."""
    match = re.search(r"\[T-\d{2}-\d{2}[A-Z]?\]", title)
    if match:
        return match.group(0)[1:-1]  # Remove brackets
    return None


def normalize_project_status(status: str) -> str:
    """Convert Project status to YAML status: 'Todo'/'In Progress'/'Done' → 'todo'/'in-progress'/'done'."""
    mapping = {
        "Todo": "todo",
        "In Progress": "in-progress",
        "Done": "done",
    }
    return mapping.get(status, "todo")


def load_tasks_config(yaml_path: Path) -> Dict:
    """Load tasks from YAML."""
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_tasks_config(yaml_path: Path, config: Dict) -> None:
    """Save tasks to YAML."""
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def update_yaml_status(config: Dict, task_id: str, new_status: str) -> bool:
    """Update task status in YAML config. Returns True if changed."""
    # Find the task in sprints
    for sprint_key, sprint_data in config.get("sprints", {}).items():
        for task_key, task_data in sprint_data.get("tasks", {}).items():
            if task_key == task_id:
                old_status = task_data.get("status", "todo")
                if old_status != new_status:
                    task_data["status"] = new_status
                    return True
                return False
    return False


def get_phase_from_task_id(task_id: str) -> Optional[int]:
    """Extract phase number from task ID like T-06-01 → 6."""
    match = re.search(r"T-(\d{2})", task_id)
    if match:
        phase_num = int(match.group(1))
        return phase_num
    return None


def get_sprint_title(config: Dict, task_id: str) -> Optional[str]:
    """Get sprint title for a task."""
    for sprint_key, sprint_data in config.get("sprints", {}).items():
        for task_key, task_data in sprint_data.get("tasks", {}).items():
            if task_key == task_id:
                return sprint_data.get("title", "")
    return None


def update_cronograma_status(md_path: Path, task_statuses: Dict[int, str]) -> bool:
    """
    Update cronograma table with current status based on task statuses.
    task_statuses: {phase_num: "todo"|"in-progress"|"done"}
    Returns True if changes were made.
    """
    if not md_path.exists():
        return False

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    changed = False

    # Map status to emoji
    status_map = {
        "done": "✅ Concluído",
        "in-progress": "🟡 Em andamento",
        "todo": "⬜ Pendente",
    }

    # For each phase, find and update the status in the table
    for phase_num, yaml_status in task_statuses.items():
        emoji_status = status_map.get(yaml_status, "⬜ Pendente")

        # Find the table row for this phase and replace status script column
        # Pattern: "| \d+.*?status script... | Status script column |"
        phase_pattern = rf"(\|\s*{phase_num}\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*)[^|]*(\s*\|)"
        replacement = rf"\1{emoji_status}\2"

        new_content = re.sub(phase_pattern, replacement, content)
        if new_content != content:
            changed = True
            content = new_content

    if changed:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)

    return changed


def sync_project_to_tasks(
    owner: str, project_number: int, repo: str, yaml_path: Path, md_path: Path
) -> None:
    """Main sync function: Project → YAML → MD."""
    print("=" * 60)
    print(f"Syncing from GitHub Project {project_number} to local files")
    print("=" * 60)
    print()

    # Load current YAML
    config = load_tasks_config(yaml_path)
    if not config:
        print("Error: Could not load tasks.yaml")
        return

    # Get Project items
    items = get_project_items(owner, project_number)
    if not items:
        print("Warning: No items found in project")
        return

    print(f"Found {len(items)} items in project")
    print()

    # Track changes
    status_changes = {}
    task_statuses = {}  # For cronograma update
    any_yaml_changes = False

    # Process each item
    for item in items:
        title = item.get("title", "")
        status = item.get("status", "Todo")

        task_id = extract_task_id(title)
        if not task_id:
            continue

        # Normalize status
        yaml_status = normalize_project_status(status)
        phase_num = get_phase_from_task_id(task_id)

        # Update YAML
        if update_yaml_status(config, task_id, yaml_status):
            status_changes[task_id] = yaml_status
            any_yaml_changes = True
            print(f"  ✓ {task_id}: {status}")
        
        # Track for cronograma
        if phase_num and yaml_status not in task_statuses:
            task_statuses[phase_num] = yaml_status

    print()

    if any_yaml_changes:
        # Save updated YAML
        save_tasks_config(yaml_path, config)
        print(f"✓ Updated {yaml_path}")
        print(f"  {len(status_changes)} task(s) status changed")
        print()

        # Update cronograma
        if update_cronograma_status(md_path, task_statuses):
            print(f"✓ Updated {md_path}")
        else:
            print(f"ℹ No cronograma changes needed")
    else:
        print("ℹ No status changes detected")

    print()
    print("=" * 60)
    print("Sync complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync GitHub Project status back to tasks.yaml and cronograma"
    )
    parser.add_argument(
        "--project-number", type=int, required=True, help="GitHub Project number"
    )
    parser.add_argument("--owner", default="jpcosta90", help="GitHub username (owner)")
    parser.add_argument(
        "--repo", default="CaVL-Doc", help="Repository name (for reference)"
    )

    project_dir = Path(__file__).parent.parent.parent  # Go to repo root
    yaml_path = project_dir / "docs" / "tasks.yaml"
    md_path = project_dir / "docs" / "cronograma_fases_pesquisa.md"

    args = parser.parse_args()

    sync_project_to_tasks(
        owner=args.owner,
        project_number=args.project_number,
        repo=args.repo,
        yaml_path=yaml_path,
        md_path=md_path,
    )
