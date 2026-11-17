# Deployment

## Production topology

- Vercel serves `frontend/` from the protected `main` branch.
- Cloud Run serves the FastAPI API in `northamerica-northeast2`.
- Artifact Registry stores immutable backend images tagged by Git commit SHA.
- A private, versioned GCS bucket stores the selected production model artifact.

## Backend release gate

The `Deploy Backend` workflow runs only after the `CI` workflow succeeds on `main`, or through a
manual dispatch from `main`. GitHub exchanges its short-lived OIDC token through a Workload
Identity Provider restricted to `Ruowen0613/anime-taste-dna` and `refs/heads/main`. No service
account key is stored in GitHub.

The workflow downloads this versioned artifact before building:

```text
gs://anime-taste-dna-models-197238956100/
  signed-rating-hybrid-v2/signed_rating_hybrid.npz
```

The Docker build fails unless the artifact matches the committed byte size, SHA-256, required
arrays, dimensions, and model version in `backend/model/signed-rating-hybrid-v2.json`. A successful
build is pushed under the source commit SHA, deployed with `min-instances=0` and `max-instances=1`,
and accepted only after the public `/health` smoke test passes.

## Deployment identities

- `github-deployer@anime-taste-dna.iam.gserviceaccount.com` can read only the model bucket, write
  only the backend image repository, update Cloud Run services, and use only the runtime identity.
- `anime-taste-dna-runtime@anime-taste-dna.iam.gserviceaccount.com` is the application runtime
  identity and has no additional project roles.
