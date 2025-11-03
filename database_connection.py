import os

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'ncs@2025'),
    'database': os.getenv('DB_NAME', 'ncs_poc'),
    'timeout': 30,
    'command_timeout': 60
}