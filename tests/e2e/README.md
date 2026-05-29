# E2E UI & Accessibility Checks

## Scope
- Visual screenshots in `light` and `dark` theme on:
  - Login
  - Chat (empty + with sources)
  - Documents
  - Dashboard
  - Settings
- Automated accessibility scan with `axe-core` (blocking on `serious` and `critical` violations).

## Commands
- `npm run test:e2e`
- `npm run test:e2e:visual`
- `npm run test:e2e:a11y`

## Runtime options
- `E2E_BASE_URL=http://127.0.0.1:3002` to target a running frontend.
- `PW_SKIP_WEBSERVER=1` to skip auto-starting the dev server.
- `A11Y_STRICT=1` to fail on both `critical` and `serious` axe violations.
  - default mode still fails on `critical`, and attaches full JSON reports.

## Reports
- HTML report: `playwright-report/index.html`
- Per-test artifacts/screenshots: `test-results/`
