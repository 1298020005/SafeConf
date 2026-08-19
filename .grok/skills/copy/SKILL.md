---
name: copy
description: Push content to Windows clipboard bridge. User runs /copy.
disable-model-invocation: true
user-invocable: true
---

# /copy

```bash
bash agents/grok/scripts/clipboard_adapter.sh
```

Output script stdout only. ASCII. Tell user: wait 1s then Ctrl+V (if clip_sync running).