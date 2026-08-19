# Copy buffer (adapter input)

After each substantive reply (except /copy skill):
- Write plain-text summary to `agents/grok/.copy_buffer.txt` (UTF-8).
- Chinese content goes ONLY into buffer/COPY.txt, never into chat for copy purposes.

User copies via `/copy` -> `COPY.txt` in editor, NOT from chat.