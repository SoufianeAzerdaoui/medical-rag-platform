# systemd deployment templates

These files are templates for a host-based deployment on DigitalOcean.

Adjust `/opt/medical-rag-platform` to your actual checkout path before installing them under `/etc/systemd/system/`.

Suggested order:

1. install Ollama
2. place `.env` at the project root
3. enable `ollama.service`
4. enable `medical-rag-backend.service`
5. enable `medical-rag-frontend.service`
