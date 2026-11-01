"""Recognize only the exact system console helper, after owned Job membership."""
from pathlib import Path

def console_command(command,system_directory):
    return (type(command) is list and len(command)==2 and command[1]=='0x4' and
            Path(command[0].removeprefix('\\??\\')).resolve()==(Path(system_directory)/'conhost.exe').resolve())
