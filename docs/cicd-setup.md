# CI/CD setup

This repository includes a minimal GitHub Actions setup:

- `.github/workflows/ci.yml`
- `.github/workflows/deploy.yml`

## What CI does

On push and pull request:

- installs Node.js dependencies
- runs `npm run typecheck`
- runs `npm run build`

## What CD does

On push to `main`:

- opens an SSH session to the DigitalOcean Droplet
- pulls the latest `main`
- rebuilds and restarts Docker containers
- reloads Caddy

## GitHub secrets to create

In `GitHub -> Settings -> Secrets and variables -> Actions`, create:

- `DO_HOST`: `107.170.16.115`
- `DO_PORT`: `22`
- `DO_USER`: `deploy`
- `DO_SSH_PRIVATE_KEY`: private SSH key used by GitHub Actions
- `DEPLOY_PATH`: `/home/deploy/apps/medical-rag-platform`

## Create a dedicated SSH key for GitHub Actions

Run on your local machine:

```bash
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/github_actions_do
```

This creates:

- private key: `~/.ssh/github_actions_do`
- public key: `~/.ssh/github_actions_do.pub`

## Add the public key on the server

Run on your server:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
```

Append the content of:

```bash
cat ~/.ssh/github_actions_do.pub
```

from your local machine, then:

```bash
chmod 600 ~/.ssh/authorized_keys
```

## Add the private key to GitHub

Copy the full content of:

```bash
cat ~/.ssh/github_actions_do
```

into the GitHub Actions secret:

- `DO_SSH_PRIVATE_KEY`

## Required sudo permission on the server

The deploy workflow runs:

```bash
sudo systemctl reload caddy
```

If `deploy` requires a password for sudo, add a sudoers rule on the server:

```bash
sudo visudo -f /etc/sudoers.d/deploy-caddy
```

Add:

```txt
deploy ALL=NOPASSWD:/usr/bin/systemctl reload caddy
```

## Notes

- If the repository is private, the server must also be able to `git pull origin main`.
- If needed, configure a read-only deploy key or HTTPS token for the repository on the server.
