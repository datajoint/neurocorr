# Local database

Run a PostgreSQL database for development and testing.

```bash
docker compose up -d        # start
docker compose ps           # status
docker compose down         # stop
docker compose down -v      # stop and remove the data
```

| Setting | Value |
|---|---|
| Host | localhost |
| Port | 5432 |
| User | datajoint |
| Password | datajoint |

`datajoint.json` at the repository root points at this database. Put the password in
`.secrets/database.password` or the `DJ_PASS` environment variable.
