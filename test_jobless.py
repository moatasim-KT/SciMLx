
import json
import subprocess
import time

def test_mcp_server():
    command = [
        "/opt/homebrew/bin/npx", "-y", "mcp-remote",
        "https://mcp.jobless.dev/mcp",
        "--header", "Authorization: Bearer T7-tG66KfGiwov66ScsO3VGRJN1ekNzvowrpiu2IuMRhRg1Yp4JlvQktAmXVqwc7"
    ]
    
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for the proxy to establish
    time.sleep(5)
    
    # Send listTools request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    
    # Read response
    try:
        # We need to skip the log lines from mcp-remote which go to stderr
        # But some might go to stdout? Usually npx output goes to stdout/stderr.
        # mcp-remote likely writes logs to stderr and standard JSON-RPC to stdout.
        response = process.stdout.readline()
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error reading response: {e}")
    finally:
        process.terminate()

if __name__ == "__main__":
    test_mcp_server()
