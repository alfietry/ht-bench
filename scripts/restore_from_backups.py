"""
Restore result files from backups to undo the broken evaluation update.
"""
import json
from pathlib import Path
from datetime import datetime
import shutil
import sys

def restore_from_backups():
    """Restore original files from backups"""
    results_dir = Path("results")
    backup_dir = results_dir / "backups"
    
    if not backup_dir.exists():
        print("❌ No backup directory found!")
        return
    
    # Find all backup files
    backup_files = list(backup_dir.glob("results_*_backup_*.json"))
    
    if not backup_files:
        print("❌ No backup files found!")
        return
    
    print(f"📁 Found {len(backup_files)} backup files")
    
    # Group backups by original filename (take most recent)
    latest_backups = {}
    for backup in backup_files:
        # Parse: results_20251213_202618_backup_20260106_104629.json
        parts = backup.stem.split('_backup_')
        if len(parts) == 2:
            original_name = parts[0] + '.json'
            timestamp = parts[1]
            
            if original_name not in latest_backups or timestamp > latest_backups[original_name][1]:
                latest_backups[original_name] = (backup, timestamp)
    
    print(f"📄 Unique originals to restore: {len(latest_backups)}")
    
    restored = 0
    failed = 0
    
    for original_name, (backup_path, timestamp) in latest_backups.items():
        original_path = results_dir / original_name
        
        try:
            # Copy backup over the current file
            shutil.copy2(backup_path, original_path)
            restored += 1
            print(f"  ✅ Restored {original_name}")
        except Exception as e:
            failed += 1
            print(f"  ❌ Failed to restore {original_name}: {e}")
    
    print()
    print("=" * 60)
    print(f"✅ Restored: {restored}")
    print(f"❌ Failed: {failed}")
    print("=" * 60)

if __name__ == "__main__":
    restore_from_backups()
