# Agent Instructions

## Notification on job completion
After every task or job that you complete, call the Pushover notification script with a concise one-line summary of what was done:

```sh
python research-utility/scripts/pushover_notify.py "<one-line summary>"
```

Keep the message brief and descriptive — e.g. `"Fixed shape mismatch in attention layer"` or `"Ran full evaluation suite, all 42 tests pass"`.

## Delta SSH login workflow
When working with the `delta` host, prefer a persistent SSH session tool such as Zed's `ssh-tmux` MCP server instead of the stateless terminal.

Working login sequence:
1. Open an SSH session to host alias `delta`.
2. Wait for the password prompt.
3. Ask the user to provide or enter the password through the live SSH session.
4. At the Duo menu, send `1` for `Duo Push`.
5. Wait for the user to approve the push notification.
6. Confirm the remote shell prompt appears before running commands.

Notes:
- Do not claim Delta access is unattended; a human must still complete MFA.
- Keep the session open while waiting for Duo approval.
- Never store passwords or MFA codes in repo files.
