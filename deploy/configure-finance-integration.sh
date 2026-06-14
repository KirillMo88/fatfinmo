#!/bin/sh
set -eu

fatfinmo_env=/opt/fatfinmo/.env
hermes_env=/opt/habit-bot/.env
secret="$(openssl rand -hex 32)"

clean_env() {
    source_file="$1"
    target_file="${source_file}.new"
    grep -v -E '^(FINANCE_API_TOKEN|FINANCE_REPORT_API_TOKEN|FINANCE_REPORT_API_URL|FINANCE_RECIPIENT_ID)=' \
        "$source_file" > "$target_file"
    chmod --reference="$source_file" "$target_file"
    mv "$target_file" "$source_file"
}

clean_env "$fatfinmo_env"
clean_env "$hermes_env"

printf 'FINANCE_API_TOKEN=%s\n' "$secret" >> "$fatfinmo_env"
printf 'FINANCE_REPORT_API_TOKEN=%s\n' "$secret" >> "$hermes_env"
printf 'FINANCE_REPORT_API_URL=http://finance-api:9000\n' >> "$hermes_env"
printf 'FINANCE_RECIPIENT_ID=45317676\n' >> "$hermes_env"

chmod 600 "$fatfinmo_env" "$hermes_env"
docker network inspect finance_internal >/dev/null 2>&1 \
    || docker network create finance_internal >/dev/null

