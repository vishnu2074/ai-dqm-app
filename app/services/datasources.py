from sqlalchemy.orm import Session
from app.models import DataSource
from app.schemas import DataSourceCreate, DataSourceTest
from cryptography.fernet import Fernet
import os
from azure.storage.blob import BlobServiceClient
try:
    import psycopg2  # optional — only needed for PostgreSQL connections
except ImportError:
    psycopg2 = None  # PostgreSQL not available; SQLite/Azure used instead
import traceback
import sys

# Encryption key — persisted to a local file so it survives restarts.
# In production, use an environment variable or a secrets manager.
_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".encryption_key")

def _load_or_create_key() -> bytes:
    key_path = os.path.normpath(_KEY_FILE)
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            return f.read().strip()
    key = Fernet.generate_key()
    with open(key_path, "wb") as f:
        f.write(key)
    return key

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", None)
if ENCRYPTION_KEY:
    ENCRYPTION_KEY = ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY
else:
    ENCRYPTION_KEY = _load_or_create_key()

cipher = Fernet(ENCRYPTION_KEY)

def encrypt_password(password: str) -> str:
    """Encrypt a plain-text secret (password, connection string, etc.)."""
    return cipher.encrypt(password.encode()).decode()

def decrypt_password(encrypted: str) -> str:
    """Decrypt an encrypted secret back to plain text."""
    return cipher.decrypt(encrypted.encode()).decode()

def _is_databricks_host(host: str) -> bool:
    """Detect if the host is a Databricks workspace or SQL endpoint."""
    if not host:
        return False
    host_lower = host.lower()
    
    # Databricks endpoint patterns
    # 1. AWS: *.cloud.databricks.com
    # 2. Azure: *.database.azuredatabricks.net (or *.azuredatabricks.net)
    # 3. GCP: *.cloud.databricks.com (same as AWS)
    # 4. Generic: databricks.com domain
    if "cloud.databricks.com" in host_lower or "databricks.com" in host_lower or "azuredatabricks.net" in host_lower:
        return True
    
    # Short SQL endpoint ID patterns (instance-*, sql-endpoint-*)
    if host_lower.startswith("instance-") or host_lower.startswith("sql-endpoint-"):
        return True
    
    return False

def _get_default_port(db_type: str, host: str = None) -> int:
    """Get the default port for a database type, considering Databricks."""
    db_type_upper = db_type.upper() if db_type else ""
    
    if db_type_upper in ("POSTGRESQL", "POSTGRES"):
        # Databricks Lakehouse uses port 5432 (with SSL enforcement)
        # Standard PostgreSQL also uses 5432
        # The difference is SSL mode, not the port
        return 5432
    elif db_type_upper == "MYSQL":
        return 3306
    return None

def _sanitize_error(msg: str) -> str:
    """Strip credentials / keys from error messages before returning to clients."""
    import re as _re
    msg = _re.sub(r'(AccountKey|SharedAccessSignature|sig|DefaultEndpointsProtocol)=[^;\s&]+', r'\1=***', msg, flags=_re.IGNORECASE)
    msg = _re.sub(r'(password|pwd|secret|token)=[^;\s&]+', r'\1=***', msg, flags=_re.IGNORECASE)
    return msg

def test_connection(dto: DataSourceTest) -> dict:
    try:
        if dto.type.upper() == "AZURE_BLOB":
            if not dto.connection_string or not dto.container_name:
                return {"success": False, "message": "Connection string and container name are required"}
            client = BlobServiceClient.from_connection_string(dto.connection_string)
            container_client = client.get_container_client(dto.container_name)
            # Test by getting container properties
            container_client.get_container_properties()
            return {"success": True, "message": "Connection successful"}
        elif dto.type.upper() in ("POSTGRESQL", "POSTGRES"):
            if not dto.host or not dto.database or not dto.username or not dto.password:
                return {"success": False, "message": "Host, database, username, and password are required"}
            
            # Normalize type
            normalized_type = "POSTGRESQL"
            
            # Auto-detect Databricks - always enforce SSL for Databricks endpoints
            is_databricks = _is_databricks_host(dto.host)
            port = dto.port or _get_default_port(normalized_type, dto.host)
            
            # For Databricks: enforce SSL "require" mode unless user explicitly overrides
            # For regular PostgreSQL: SSL is optional
            if is_databricks:
                ssl_mode = dto.ssl_mode if dto.ssl_mode is not None else "require"
            else:
                ssl_mode = dto.ssl_mode
            
            connect_kwargs = dict(
                host=dto.host,
                port=port,
                database=dto.database,
                user=dto.username,
                password=dto.password,
                connect_timeout=15,
            )
            if ssl_mode:
                connect_kwargs["sslmode"] = ssl_mode
            
            # DEBUG: Log connection attempt
            print(f"[DEBUG] Attempting PostgreSQL connection:", file=sys.stderr)
            print(f"[DEBUG] Host: {dto.host}, Port: {port}, DB: {dto.database}, User: {dto.username}, SSL: {ssl_mode}", file=sys.stderr)
            print(f"[DEBUG] Is Databricks: {is_databricks}", file=sys.stderr)
            
            conn = psycopg2.connect(**connect_kwargs)
            conn.close()
            return {"success": True, "message": "Connection successful"}
        else:
            return {"success": False, "message": f"Unsupported type: {dto.type}"}
    except psycopg2.OperationalError as e:
        error_msg = str(e)
        # DEBUG: Log full error
        print(f"[ERROR] psycopg2.OperationalError: {error_msg}", file=sys.stderr)
        print(f"[ERROR] Full traceback:\n{traceback.format_exc()}", file=sys.stderr)
        
        # Provide specific error messages for common issues
        if "connection refused" in error_msg.lower():
            msg = "Connection refused - verify host, port, and that the server is running"
        elif "could not translate host name" in error_msg.lower() or "getaddrinfo failed" in error_msg.lower():
            msg = "Cannot resolve hostname - verify the endpoint address is correct and reachable"
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            msg = "Connection timeout - endpoint may be unreachable, offline, or network is blocking the connection"
        elif "password authentication failed" in error_msg.lower():
            msg = "Authentication failed - verify username and password are correct"
        elif "FATAL" in error_msg and "role" in error_msg.lower():
            msg = "Role/user does not exist or lacks permissions - verify username with your administrator"
        elif "ssl" in error_msg.lower() or "certificate" in error_msg.lower():
            msg = "SSL/certificate error - try disabling SSL or updating certificate settings" if not _is_databricks_host(dto.host) else "SSL error with Databricks - ensure SSL mode is set to 'require'"
        else:
            msg = _sanitize_error(error_msg)
        return {"success": False, "message": f"Connection failed: {msg}"}
    except Exception as e:
        error_msg = str(e)
        # DEBUG: Log full error
        print(f"[ERROR] General Exception: {type(e).__name__}: {error_msg}", file=sys.stderr)
        print(f"[ERROR] Full traceback:\n{traceback.format_exc()}", file=sys.stderr)
        
        msg = _sanitize_error(error_msg)
        return {"success": False, "message": f"Connection failed: {msg}"}

def create_datasource(db: Session, dto: DataSourceCreate) -> DataSource:
    # Check unique name
    existing = db.query(DataSource).filter(DataSource.name == dto.name).first()
    if existing:
        raise ValueError("Data source name already exists")

    # Normalize type early
    normalized_type = dto.type.upper()
    if normalized_type in ("POSTGRESQL", "POSTGRES"):
        normalized_type = "POSTGRESQL"
    elif normalized_type == "MYSQL":
        normalized_type = "MYSQL"

    # Auto-detect Databricks
    is_databricks = (normalized_type == "POSTGRESQL" and dto.host and _is_databricks_host(dto.host))
    
    # Normalize port with context-aware default
    normalized_port = dto.port or _get_default_port(normalized_type, dto.host)
    
    # Auto-enforce SSL for Databricks if not explicitly set
    # For Databricks: default to "require" SSL mode
    # For regular PostgreSQL: SSL optional
    if is_databricks:
        normalized_ssl_mode = dto.ssl_mode if dto.ssl_mode is not None else "require"
    else:
        normalized_ssl_mode = dto.ssl_mode

    # Check duplicate connection (same target, different name)
    if normalized_type in ("POSTGRESQL", "MYSQL"):
        dup = (
            db.query(DataSource)
            .filter(
                DataSource.type == normalized_type,
                DataSource.host == dto.host,
                DataSource.port == normalized_port,
                DataSource.database == dto.database,
            )
            .first()
        )
        if dup:
            raise ValueError(
                f"A data source already connects to {dto.host}:{normalized_port}/{dto.database} "
                f"(existing source: '{dup.name}')"
            )
    elif normalized_type == "AZURE_BLOB":
        # For Azure, check by container name + name pattern (connection string is encrypted)
        dup = (
            db.query(DataSource)
            .filter(
                DataSource.type == "AZURE_BLOB",
                DataSource.container_name == dto.container_name,
            )
            .first()
        )
        if dup:
            raise ValueError(
                f"A data source already connects to this Azure Blob container "
                f"(existing source: '{dup.name}')"
            )
    elif normalized_type == "S3":
        dup = (
            db.query(DataSource)
            .filter(
                DataSource.type == "S3",
                DataSource.host == dto.host,
            )
            .first()
        )
        if dup:
            raise ValueError(
                f"A data source already connects to this S3 bucket "
                f"(existing source: '{dup.name}')"
            )
    elif normalized_type == "REST API":
        dup = (
            db.query(DataSource)
            .filter(
                DataSource.type == "REST API",
                DataSource.host == dto.host,
            )
            .first()
        )
        if dup:
            raise ValueError(
                f"A data source already connects to this API endpoint "
                f"(existing source: '{dup.name}')"
            )

    # Test connection with normalized values
    test_dto = DataSourceTest(
        type=normalized_type,
        connection_string=dto.connection_string,
        container_name=dto.container_name,
        host=dto.host,
        port=normalized_port,
        database=dto.database,
        username=dto.username,
        password=dto.password,
        ssl_mode=normalized_ssl_mode
    )
    test_result = test_connection(test_dto)
    if not test_result["success"]:
        raise ValueError(test_result["message"])

    # Encrypt secrets before storing
    encrypted_password = encrypt_password(dto.password) if dto.password else None
    encrypted_conn_str = encrypt_password(dto.connection_string) if dto.connection_string else None

    ds = DataSource(
        name=dto.name,
        type=normalized_type,
        host=dto.host,
        port=normalized_port,
        database=dto.database,
        username=dto.username,
        encrypted_password=encrypted_password,
        connection_string=encrypted_conn_str,
        container_name=dto.container_name,
        ssl_mode=normalized_ssl_mode,
        status="ACTIVE",
        owner="USER"
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds

def get_datasources(db: Session):
    return db.query(DataSource).all()

def get_datasource_by_id(db: Session, id: int):
    return db.query(DataSource).filter(DataSource.id == id).first()

def delete_datasource(db: Session, id: int):
    ds = get_datasource_by_id(db, id)
    if not ds:
        raise ValueError("Data source not found")
    db.delete(ds)
    db.commit()

def toggle_datasource(db: Session, id: int):
    ds = get_datasource_by_id(db, id)
    if not ds:
        raise ValueError("Data source not found")
    ds.status = "INACTIVE" if ds.status == "ACTIVE" else "ACTIVE"
    db.commit()
    db.refresh(ds)
    return ds

def list_physical_datasets(db: Session, datasource_id: int):
    ds = get_datasource_by_id(db, datasource_id)
    if not ds:
        raise ValueError("Data source not found")

    if ds.type == "AZURE_BLOB":
        try:
            conn_str = decrypt_password(ds.connection_string) if ds.connection_string else None
            if not conn_str:
                raise ValueError("No connection string stored for this data source")
            client = BlobServiceClient.from_connection_string(conn_str)
            container_client = client.get_container_client(ds.container_name)
            blobs = container_client.list_blobs()
            
            # Filter to only actual data files (not directories or intermediate paths)
            data_extensions = ('.csv', '.xlsx', '.json', '.parquet', '.xls', '.txt', '.tsv')
            files = []
            for blob in blobs:
                # Skip virtual directories (paths ending with /)
                if blob.name.endswith('/'):
                    continue
                # Only include files with data extensions
                if any(blob.name.lower().endswith(ext) for ext in data_extensions):
                    files.append({"name": blob.name})
            
            return files
        except Exception as e:
            msg = _sanitize_error(str(e))
            raise ValueError(f"Failed to list blobs: {msg}")
    elif ds.type == "POSTGRESQL":
        try:
            password = decrypt_password(ds.encrypted_password) if ds.encrypted_password else None
            connect_kwargs = dict(
                host=ds.host,
                port=ds.port,
                database=ds.database,
                user=ds.username,
                password=password,
            )
            if ds.ssl_mode:
                connect_kwargs["sslmode"] = ds.ssl_mode
            conn = psycopg2.connect(**connect_kwargs)
            cursor = conn.cursor()
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = cursor.fetchall()
            conn.close()
            return [{"name": row[0]} for row in tables]
        except Exception as e:
            msg = _sanitize_error(str(e))
            raise ValueError(f"Failed to list tables: {msg}")
    else:
        return []