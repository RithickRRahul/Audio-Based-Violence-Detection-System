import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """
    Custom formatter to output machine-readable JSON logs for production observability.
    Captures timestamp, level, message, and any extra structured context.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
        }
        
        # Merge any extra kwargs passed to the logger
        if hasattr(record, "extra_info") and isinstance(record.extra_info, dict):
            log_obj.update(record.extra_info)
            
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured structured JSON logger.
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist to avoid double logging
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
    return logger
