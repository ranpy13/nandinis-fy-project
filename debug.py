import debugpy

# Allow other computers to attach (for Docker use 0.0.0.0)
debugpy.listen(("0.0.0.0", 5678))

print("⏳ Waiting for debugger attach...")
debugpy.wait_for_client()  # Optional: pause until debugger attaches
print("✅ Debugger is attached!")
