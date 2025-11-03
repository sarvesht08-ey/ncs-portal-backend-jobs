# check_server_env.py
import os
import asyncio
import sys

# Add your project to path if needed
sys.path.insert(0, '/var/www/ncs-portal-backend-jobs')

async def diagnose_connection():
    """Comprehensive connection diagnostics"""
    
    print("=== Database Connection Diagnostics ===\n")
    
    # 1. Check environment variable
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not set in environment")
        print("\n💡 Try loading from .env file...")
        
        # Try loading from .env
        try:
            from dotenv import load_dotenv
            load_dotenv()
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                print("✓ Loaded DATABASE_URL from .env file")
            else:
                print("❌ DATABASE_URL not found in .env either")
                return
        except ImportError:
            print("❌ python-dotenv not installed")
            return
    else:
        print("✓ DATABASE_URL is set in environment")
    
    # 2. Parse connection string
    from urllib.parse import urlparse
    try:
        parsed = urlparse(db_url)
        print(f"\n📋 Connection Details:")
        print(f"  Scheme: {parsed.scheme}")
        print(f"  Host: {parsed.hostname}")
        print(f"  Port: {parsed.port or 5432}")
        print(f"  Database: {parsed.path.lstrip('/')}")
        print(f"  Username: {parsed.username}")
        print(f"  Has Password: {'Yes' if parsed.password else 'No'}")
        print(f"  SSL Mode: {'Yes' if 'sslmode' in db_url else 'No (MISSING - Azure needs this!)'}")
        if parsed.query:
            print(f"  Query Params: {parsed.query}")
    except Exception as e:
        print(f"❌ Failed to parse DATABASE_URL: {e}")
        return
    
    # 3. DNS resolution
    print(f"\n🌐 Testing DNS resolution for {parsed.hostname}...")
    import socket
    try:
        ip = socket.gethostbyname(parsed.hostname)
        print(f"✓ Hostname resolves to: {ip}")
    except socket.gaierror as e:
        print(f"❌ DNS resolution failed: {e}")
        print("   This means the hostname cannot be found. Check:")
        print("   - Is the hostname spelled correctly?")
        print("   - Is your server's DNS configured?")
        return
    
    # 4. TCP connection
    port = parsed.port or 5432
    print(f"\n🔌 Testing TCP connection to {parsed.hostname}:{port}...")
    try:
        sock = socket.create_connection((parsed.hostname, port), timeout=10)
        sock.close()
        print(f"✓ TCP connection successful")
    except socket.timeout:
        print(f"❌ TCP connection timed out")
        print("   Check: Server firewall, Azure firewall rules, network policies")
        return
    except ConnectionRefusedError:
        print(f"❌ TCP connection refused")
        print("   Check: PostgreSQL is running, port is correct, firewall allows connections")
        return
    except Exception as e:
        print(f"❌ TCP connection failed: {type(e).__name__}: {e}")
        return
    
    # 5. Check if asyncpg is installed
    try:
        import asyncpg
        print(f"\n📦 asyncpg version: {asyncpg.__version__}")
    except ImportError:
        print(f"\n❌ asyncpg not installed. Install with: pip install asyncpg")
        return
    
    # 6. PostgreSQL authentication
    print(f"\n🔐 Testing PostgreSQL connection...")
    try:
        conn = await asyncpg.connect(db_url, timeout=30)
        version = await conn.fetchval("SELECT version()")
        print(f"✓ PostgreSQL connection successful!")
        print(f"  Version: {version[:100]}")
        
        # Test a simple query
        result = await conn.fetchval("SELECT 1")
        print(f"✓ Simple query test passed (result: {result})")
        
        await conn.close()
        
    except asyncpg.InvalidPasswordError as e:
        print(f"❌ Authentication failed")
        print(f"   Error: {e}")
        print("   Check: Username and password are correct")
        if ".azure." in parsed.hostname and "@" not in parsed.username:
            print(f"   💡 Azure PostgreSQL requires username format: username@servername")
            server_name = parsed.hostname.split('.')[0]
            print(f"   Try: {parsed.username}@{server_name}")
        return
        
    except asyncpg.InvalidCatalogNameError as e:
        print(f"❌ Database does not exist")
        print(f"   Error: {e}")
        print(f"   The database '{parsed.path.lstrip('/')}' was not found on the server")
        return
        
    except asyncpg.CannotConnectNowError as e:
        print(f"❌ Database is not accepting connections")
        print(f"   Error: {e}")
        print("   The database server may be starting up or in maintenance mode")
        return
        
    except Exception as e:
        print(f"❌ Connection failed: {type(e).__name__}")
        print(f"   Error: {e}")
        
        # Provide helpful hints based on error
        error_str = str(e).lower()
        if "ssl" in error_str or "certificate" in error_str:
            print("\n   💡 SSL Issue detected. For Azure PostgreSQL, add to connection string:")
            print("      ?sslmode=require")
        elif "timeout" in error_str:
            print("\n   💡 Connection timeout. Check:")
            print("      - Firewall rules (Azure PostgreSQL → Networking → Firewall)")
            print("      - Network connectivity from this server")
        
        return
    
    # 7. Test actual table query
    print(f"\n📊 Testing vacancies_summary table...")
    try:
        conn = await asyncpg.connect(db_url, timeout=30)
        count = await conn.fetchval("SELECT COUNT(*) FROM vacancies_summary")
        print(f"✓ Query successful - Found {count:,} jobs in database")
        
        # Get a sample row
        sample = await conn.fetchrow("SELECT ncspjobid, title FROM vacancies_summary LIMIT 1")
        if sample:
            print(f"✓ Sample job: {sample['ncspjobid']} - {sample['title']}")
        
        await conn.close()
        
    except asyncpg.UndefinedTableError:
        print(f"❌ Table 'vacancies_summary' does not exist")
        print("   Check: Is this the correct database? Have migrations run?")
    except Exception as e:
        print(f"❌ Query failed: {type(e).__name__}: {e}")
    
    print("\n" + "="*50)
    print("✅ All diagnostics completed!")
    print("="*50 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(diagnose_connection())
    except KeyboardInterrupt:
        print("\n\nDiagnostics interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()