# Deployment safety

- On the fatfinmo VPS, `/opt/fatfinmo/current` points to the active release. Run Docker Compose commands from that directory, using the `fatfinmo` project and `/opt/fatfinmo/.env`.
- Do not run Compose from `/opt/fatfinmo` or use `/opt/fatfinmo/docker-compose.yml`: that root checkout and Compose file are legacy and may restore an older application/configuration.
- Before any VPS deployment, verify `readlink -f /opt/fatfinmo/current` and the running containers' `com.docker.compose.project.config_files` label. If they disagree, investigate before changing services.
