## Expose FastAPI with ngrok

1) Put your token in `.env`:

```
NGROK_AUTH_TOKEN=...
```

2) Run:

```powershell
.\run_fastapi_ngrok.ps1
```

It prints an `https://...ngrok...` public URL you can share.

If you need a fixed URL, you must use an ngrok reserved domain and run:

```powershell
.\run_fastapi_ngrok.ps1 -Domain your-reserved-domain.example.com
```
