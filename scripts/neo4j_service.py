from __future__ import annotations

import subprocess
import time

from neo4j import GraphDatabase


def wait_for_bolt(uri: str, database: str, timeout_s: int = 60) -> None:
    print("  Waiting for Neo4j Bolt endpoint...", flush=True)
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        driver = None
        try:
            driver = GraphDatabase.driver(uri, auth=None)
            with driver.session(database=database) as session:
                session.run("RETURN 1").consume()
            return
        except Exception as e:
            last_error = e
            time.sleep(1)
        finally:
            if driver is not None:
                driver.close()

    raise RuntimeError(f"Neo4j did not become ready at {uri}: {last_error}")


def ensure_running(uri: str, database: str) -> None:
    result = subprocess.run(
        ["neo4j", "status"], capture_output=True, text=True, check=False
    )
    if "Neo4j is running" not in f"{result.stdout}\n{result.stderr}":
        print("  Starting Neo4j...", flush=True)
        subprocess.run(["sudo", "neo4j", "start"], check=True)
    wait_for_bolt(uri, database)


def restart(uri: str, database: str) -> None:
    subprocess.run(["sudo", "neo4j", "stop"], check=False)
    subprocess.run(["sudo", "neo4j", "start"], check=True)
    wait_for_bolt(uri, database)


def stop() -> None:
    subprocess.run(["sudo", "neo4j", "stop"], check=False)
