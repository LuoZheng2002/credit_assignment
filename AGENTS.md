# Agent Instructions

## Notification on job completion
After every task or job that you complete, call the Pushover notification script with a concise one-line summary of what was done:

```sh
python research-utility/scripts/pushover_notify.py "<one-line summary>"
```

Keep the message brief and descriptive — e.g. `"Fixed shape mismatch in attention layer"` or `"Ran full evaluation suite, all 42 tests pass"`.
