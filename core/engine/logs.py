# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : Claude
# @Desc    : Simple colored logger with file output

import os
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Optional, TextIO, Union

class Colors:
    """Terminal color codes for different log levels"""
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    LIGHT_BLUE = '\033[94m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class LogLevel(Enum):
    """Log levels with corresponding colors"""
    DEBUG = (10, Colors.BLUE)
    OPTIMIZE = (25, Colors.CYAN)
    INFO = (20, Colors.GREEN)
    WARNING = (30, Colors.YELLOW)
    ERROR = (40, Colors.RED)
    CRITICAL = (50, Colors.MAGENTA)

class SimpleLogger:
    """Simple logger class that supports both colored terminal output and file logging"""
    
    def __init__(
        self, 
        name: str = "AutoEnv",
        log_level: Union[int, LogLevel] = LogLevel.INFO,
        log_file: Optional[str] = None,
        log_dir: str = "workspace/logs",
        console_output: bool = True
    ):
        """
        Initialize the Logger
        
        Args:
            name: Logger name
            log_level: Minimum log level to display
            log_file: Log file name (if None, will use name_YYYY-MM-DD.log)
            log_dir: Directory to store log files
            console_output: Whether to output logs to console
        """
        self.name = name
        
        # Convert LogLevel enum to int if needed
        if isinstance(log_level, LogLevel):
            self.log_level = log_level.value[0]
        else:
            self.log_level = log_level
        
        self.console_output = console_output
        self.file_output = None
        
        # Define display names for log levels
        self.level_display_names = {
            LogLevel.DEBUG: "DEBUG",
            LogLevel.OPTIMIZE: "OPTIMIZE", 
            LogLevel.INFO: "INFO",
            LogLevel.WARNING: "WARNING",
            LogLevel.ERROR: "ERROR",
            LogLevel.CRITICAL: "CRITICAL"
        }
        
        # Set up file logging
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate default log filename if not provided
            if log_file is None:
                current_date = datetime.now().strftime("%Y-%m-%d")
                log_file = f"{name}_{current_date}.log"
            
            file_path = os.path.join(log_dir, log_file)
            self.file_output = open(file_path, 'a', encoding='utf-8')
    
    def _log(self, level: LogLevel, message: str) -> None:
        """Internal method to log messages at specified level"""
        if level.value[0] < self.log_level:
            return
            
        # Format the log message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = self.level_display_names.get(level, level.name)
        formatted_msg = f"{timestamp} - {level_name} - {message}"
        
        # Write to console if enabled
        if self.console_output:
            color = level.value[1]
            # Add bold to critical messages
            if level == LogLevel.CRITICAL:
                colored_msg = f"{Colors.BOLD}{color}{formatted_msg}{Colors.RESET}"
            else:
                colored_msg = f"{color}{formatted_msg}{Colors.RESET}"
            print(colored_msg)
        
        # Write to file if enabled
        if self.file_output:
            self.file_output.write(formatted_msg + "\n")
            self.file_output.flush()
    
    def log_to_file(self, level: LogLevel, message: str) -> None:
        """
        Log a message to file only, without printing to console
        
        Args:
            level: Log level
            message: Message to log
        """
        if level.value[0] < self.log_level:
            return
            
        # Format the log message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = self.level_display_names.get(level, level.name)
        formatted_msg = f"{timestamp} - {level_name} - {message}"
        
        # Write to file if enabled
        if self.file_output:
            self.file_output.write(formatted_msg + "\n")
            self.file_output.flush()
    
    def debug(self, message: str) -> None:
        """Log a debug message"""
        self._log(LogLevel.DEBUG, message)
    
    def info(self, message: str) -> None:
        """Log an info message"""
        self._log(LogLevel.INFO, message)
    
    def optimize(self, message: str) -> None:
        """Log an optimization info message"""
        self._log(LogLevel.OPTIMIZE, message)
    
    def warning(self, message: str) -> None:
        """Log a warning message"""
        self._log(LogLevel.WARNING, message)
    
    def error(self, message: str) -> None:
        """Log an error message"""
        self._log(LogLevel.ERROR, message)
    
    def critical(self, message: str) -> None:
        """Log a critical message"""
        self._log(LogLevel.CRITICAL, message)
    
    def agent_action(self, message: str) -> None:
        """Log an agent action with special cyan color and bold formatting"""
        if self.log_level <= LogLevel.INFO.value[0]:  # Only log if INFO level or lower
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_msg = f"{timestamp} - AGENT_ACTION - {message}"
            
            # Write to console if enabled
            if self.console_output:
                colored_msg = f"{Colors.BOLD}{Colors.CYAN}{formatted_msg}{Colors.RESET}"
                print(colored_msg)
            
            # Write to file if enabled
            if self.file_output:
                self.file_output.write(formatted_msg + "\n")
                self.file_output.flush()

    def action(self, message: str) -> None:
        """Log a generic action entry with light blue color"""
        if self.log_level <= LogLevel.INFO.value[0]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_msg = f"{timestamp} - ACTION - {message}"

            if self.console_output:
                colored_msg = f"{Colors.LIGHT_BLUE}{formatted_msg}{Colors.RESET}"
                print(colored_msg)

            if self.file_output:
                self.file_output.write(formatted_msg + "\n")
                self.file_output.flush()

    def agent_thinking(self, message: str) -> None:
        """Log an agent thinking message with special white color and bold formatting"""
        if self.log_level <= LogLevel.INFO.value[0]:  # Only log if INFO level or lower
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_msg = f"{timestamp} - AGENT_THINKING - {message}"

            # Write to console if enabled
            if self.console_output:
                colored_msg = f"{Colors.BOLD}{Colors.WHITE}{formatted_msg}{Colors.RESET}"
                print(colored_msg)

            # Write to file if enabled
            if self.file_output:
                self.file_output.write(formatted_msg + "\n")
                self.file_output.flush()
    
    def __del__(self):
        """Close file handle when logger is destroyed"""
        if self.file_output:
            self.file_output.close()

# Create a singleton instance for easy import
logger = SimpleLogger()
