#!/usr/bin/env python3
"""Run this once to securely write your API keys to .env"""
import getpass
import re

env_path = "/home/user/PersonalizedJobDiscovery/.env"

print("Enter your API keys (input is hidden):\n")

anthropic_key = getpass.getpass("ANTHROPIC_API_KEY: ")
adzuna_id     = getpass.getpass("ADZUNA_APP_ID: ")

with open(env_path, "r") as f:
    content = f.read()

content = re.sub(r"ANTHROPIC_API_KEY=.*", f"ANTHROPIC_API_KEY={anthropic_key}", content)
content = re.sub(r"ADZUNA_APP_ID=.*",     f"ADZUNA_APP_ID={adzuna_id}",         content)

with open(env_path, "w") as f:
    f.write(content)

print("\nKeys saved to .env successfully.")
