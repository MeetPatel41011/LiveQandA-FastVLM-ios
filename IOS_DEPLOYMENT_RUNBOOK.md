# iOS Deployment Runbook (Mac-free, GitHub Actions + fastlane)

This runbook is the human-side companion to the in-repo CI/CD code. The
plan covers **Phase 0a–0f** (everything you must click on in browsers
because no API does it for you). Work through it top-to-bottom on a
fresh setup; each section is idempotent.

> **You are doing this once.** After Phase 0 is green, every future code
> change ships to your iPhone via `git push` alone.

---

## Phase 0a — Repo hygiene before going public

The Tavily key in [`ml-fastvlm/.env`](ml-fastvlm/.env) was committed to your
working tree. The file *is* in [`.gitignore`](ml-fastvlm/.gitignore) (line
105), so it should never have been pushed to a remote. Verify, rotate, and
move on.

1. **Verify the file was never pushed.**

   ```powershell
   cd d:\Apple5\JMKmeetUM\FastVLM
   git log --all --source -- ml-fastvlm/.env
   ```

   - Empty output → the key is local only. Skip step 2 (no rotation
     needed) but still complete steps 3-5.
   - Any commits listed → the key was pushed at some point. **Rotate
     before going public.**

2. **Rotate the Tavily key** (only if step 1 found commits):
   - Sign in at <https://app.tavily.com/>.
   - "API Keys" → revoke the old key → create a new one.
   - Update your local `ml-fastvlm/.env` with the new key (still
     gitignored, never committed).
   - If the old key was already in remote history, run
     `git filter-repo --path ml-fastvlm/.env --invert-paths` and force-push
     before going public. (`git filter-repo` is the modern replacement
     for `git filter-branch`; `pip install git-filter-repo`.)

3. **Stop tracking `.env` even by accident.**

   ```powershell
   git rm --cached ml-fastvlm/.env 2>$null
   git status   # should show "deleted: ml-fastvlm/.env" only if it was tracked
   ```

4. **Move the Tavily key to GitHub Secrets** (we'll reference it in
   Phase 0d). Tavily is read at app launch from `Info.plist`, which the
   workflow injects from the secret at build time — the file is never
   on disk on the runner.

5. **Commit the runbook + cleanup** (we'll batch this at the end of
   Phase 0).

## Phase 0b — Make the GitHub repo public

GitHub Actions `macos-latest` (and `macos-14`) minutes are billed at
**10x** the rate of Linux runners. Free private-repo plans give you 2,000
linux-min/month → effectively **200 macOS min/month** (≈10 builds). On
**public** repos, all minutes are unlimited and free. So we go public.

1. Make sure Phase 0a is done (no live secrets in tracked files).
2. **Push the current branch to GitHub** if you haven't already
   (Settings → Repositories → New repository, then `git remote add` and
   `git push`).
3. **Settings → General → Danger Zone → "Change visibility" → Public.**
4. Confirm the warning prompt.
5. Verify by visiting the repo URL in an incognito window — the README
   should load without auth.

## Phase 0c — fastlane scaffolding

Already done in-repo. See:

- [`ml-fastvlm/app/Gemfile`](ml-fastvlm/app/Gemfile) — pins fastlane.
- [`ml-fastvlm/app/fastlane/Fastfile`](ml-fastvlm/app/fastlane/Fastfile)
  — `beta` lane (build + TestFlight upload).
- [`ml-fastvlm/app/fastlane/Matchfile`](ml-fastvlm/app/fastlane/Matchfile)
  — points at the private cert repo you'll create in Phase 0d.
- [`ml-fastvlm/app/fastlane/Appfile`](ml-fastvlm/app/fastlane/Appfile)
  — reads bundle ID, Apple ID, and team ID from **environment variables**
  (set in GitHub Actions from **Secrets**). Do **not** paste real values
  into this file in the repo.

**Do not edit `Appfile` or `Matchfile` in git.** They use `ENV["..."]`
with safe placeholders as fallbacks (`com.example.iris`, `you@example.com`).
CI injects the real values from the **Secrets** table in Phase 0d below.

| Value | Where it comes from in CI |
| --- | --- |
| Bundle ID | Secret `APP_BUNDLE_IDENTIFIER` → `ENV["APP_BUNDLE_IDENTIFIER"]` |
| Apple ID email | Secret `APPLE_ID` → `ENV["APPLE_ID"]` (used by `match` username) |
| Team ID | Secret `APPLE_TEAM_ID` → `ENV["APPLE_TEAM_ID"]` |
| Private cert repo URL | Secret `MATCH_GIT_URL` → `ENV["MATCH_GIT_URL"]` |

The bundle ID `com.apple.ml.FastVLM` baked into [`project.pbxproj`](ml-fastvlm/app/FastVLM.xcodeproj/project.pbxproj)
**must** be overridden because no non-Apple team can sign with that
prefix. The fastlane `Fastfile` injects the override at build time via
`xcargs` so we don't need to edit the pbxproj.

## Phase 0d — Apple credentials + GitHub Secrets

You need Apple-side accounts/keys and **GitHub Actions secrets** (no
credentials committed in the repo).

### 1. Register the bundle ID with Apple

1. <https://developer.apple.com/account/resources/identifiers/list>
2. "+" → "App IDs" → "App"
3. Description: `Iris OS`. Bundle ID: explicit, e.g. `com.meetpatel.iris`.
4. Capabilities: enable **Speech Recognition** and **Microphone** (the
   regular `Camera` capability is already implied by the `NSCameraUsageDescription`
   key).
5. Continue → Register.

### 2. Create the App Store Connect record

1. <https://appstoreconnect.apple.com/apps>
2. "+" → "New App". Platform: iOS. Name: `Iris OS`. Bundle ID: select the
   one you just registered. SKU: anything unique, e.g. `iris-ios-001`.
3. Don't fill in marketing pages — TestFlight builds work without them.

### 3. Create the private cert repo (for fastlane match)

1. <https://github.com/new> → name: `iris-certs`. **Private**. Empty
   (no README).
2. Generate a Personal Access Token (classic) at <https://github.com/settings/tokens>:
   - Scope: `repo` (full).
   - Expiration: at least 90 days; you'll re-enter this every rotation.
   - Save the token string.
3. Compute the basic-auth header value (PowerShell):

   ```powershell
   $pat = "ghp_yourtokenhere"
   $user = "yourgithubusername"
   $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("${user}:${pat}"))
   $b64
   ```

4. Save the printed string — that's `MATCH_GIT_BASIC_AUTHORIZATION`.

### 4. Generate an App Store Connect API key

1. <https://appstoreconnect.apple.com/access/integrations/api>
2. "+" → name: `iris-ci`. Access: **App Manager**.
3. Download the `.p8` file once (cannot redownload). Note:
   - `Issuer ID` (top of the page).
   - `Key ID` (10-char string next to the new key).
4. Open the `.p8` file in Notepad. Copy its full contents (including
   `-----BEGIN PRIVATE KEY-----` lines).

### 5. Add GitHub Secrets

Settings → Secrets and variables → Actions → New repository secret. Add
every **Required** row below. `TAVILY_API_KEY` is optional.

| Secret name | Value | Required? |
| --- | --- | --- |
| `APPLE_ID` | The **Apple ID email** you use for the Developer Program / App Store Connect (same account that owns the app). Used by **fastlane match** as `username`. | **Required** for `match` on CI |
| `MATCH_PASSWORD` | A new long passphrase you invent. Save it in your password manager. Used to encrypt the cert repo. | Required |
| `MATCH_GIT_URL` | HTTPS URL of `iris-certs` (e.g. `https://github.com/you/iris-certs.git`) | Required |
| `MATCH_GIT_BASIC_AUTHORIZATION` | The base64 string from Phase 0d step 3.4 | Required |
| `APP_STORE_CONNECT_KEY_ID` | The 10-char Key ID from step 4.3 | Required |
| `APP_STORE_CONNECT_ISSUER_ID` | The Issuer ID from step 4.3 | Required |
| `APP_STORE_CONNECT_KEY` | Full contents of the `.p8` file (including the `-----BEGIN/END` lines) | Required |
| `APP_BUNDLE_IDENTIFIER` | Your chosen bundle ID, e.g. `com.meetpatel.iris` | Required |
| `APPLE_TEAM_ID` | Your 10-char Team ID | Required |
| `TAVILY_API_KEY` | The new Tavily key from Phase 0a | Optional (search still works without it via Wikipedia) |

## Phase 0e — The workflow

Already done in-repo at
[`.github/workflows/ios-build.yml`](.github/workflows/ios-build.yml).
On every push to `main` it:

1. Boots a `macos-14` runner.
2. Caches Xcode's DerivedData and the SwiftPM cache.
3. Caches the 2 GB FastVLM-1.5B-INT8 weights (keyed on the SHA of
   `get_pretrained_mlx_model.sh` — the cache only invalidates if Apple
   updates the script).
4. Runs `bash app/get_pretrained_mlx_model.sh --model 1.5b
   --dest app/FastVLM/model` if the cache missed.
5. Calls `bundle exec fastlane beta`, which uses `match` to install the
   signing cert, `gym` to build a signed `.ipa`, and `pilot` to upload
   to TestFlight.

Trigger a build manually the first time:

1. <https://github.com/you/your-repo/actions> → "iOS build" → "Run
   workflow" → branch: `main`.
2. Wait ~12-15 minutes for the first run (subsequent runs are ~6 min
   thanks to caching).

## Phase 0f — First TestFlight install

The very first build to a fresh App Store Connect record takes a one-time
Apple review (≈24 h). All builds after that are instant.

1. Once the workflow's `pilot upload` step succeeds, you'll see the
   build in TestFlight under "Builds" with status "Waiting for Review".
2. Wait for the email "Your beta app has completed beta review".
3. Add yourself as an Internal Tester at <https://appstoreconnect.apple.com>
   → My Apps → Iris OS → TestFlight → Internal Testing → "+ New Group" →
   add your Apple ID.
4. On your iPhone: install the **TestFlight** app from the App Store,
   sign in with the same Apple ID, and the Iris OS build will appear.
5. Tap "Install".

From this point onwards: `git push` → ~12 min → tap "Install" on
TestFlight → app updates on iPhone. No browser, no Mac, no friction.

---

## Operational notes

### When something goes wrong

- **`match` says "Couldn't access repository"** → `MATCH_GIT_BASIC_AUTHORIZATION`
  is wrong (re-encode the PAT), or the PAT expired, or the cert repo URL
  is wrong.
- **`gym` says "No code signing identity found"** → `MATCH_PASSWORD`
  doesn't match the encrypted repo's actual passphrase. If you forget the
  passphrase, run `bundle exec fastlane match nuke distribution` from a
  Mac (or as a one-off GH Actions job — there's a manual workflow file
  you can add) and start fresh.
- **`pilot upload` fails with 401** → the App Store Connect API key was
  rotated or has insufficient access. Re-issue at the integrations page.
- **TestFlight beta review takes >48 h** → email Apple via the App Store
  Connect "Resolution Center". This is rare for the very first build.

### When you want to add another tester

App Store Connect → My Apps → Iris OS → TestFlight → Internal Testing →
add their Apple ID. They'll get an email invite, install TestFlight,
done. Up to 100 internal testers, no review per-build.

### When you want to ship to the App Store proper

Out of scope for now. The `beta` lane in `Fastfile` only ships to
TestFlight. To go public you'd add a `release` lane that calls
`deliver` instead of `pilot`, plus all the App Store metadata. Park it.

## Phase 4b — Tuning the four Sentinel knobs on a real iPhone

Once the app is on your device via TestFlight, the *only* good way to
tune Sentinel thresholds is empirically on your actual lighting + handset.
Defaults work but are conservative.

1. **Enable PerfHUD.** Long-press anywhere on the camera preview for
   ~0.8 s. The overlay shows live `Sharp`, `Motion`, `Stable`, `New`,
   `Cooldown`, and the most recent `TTFT`.

2. **Capture three reference samples**, ~10 s each:
   - **Idle**: empty room, phone on a desk. Note the resting `Motion`
     value (typically 0.3-1.5).
   - **Held still**: pick up the phone, point at a piece of paper, hold
     normally without trying to be still. Note `Motion` (typically
     2.5-5.0).
   - **Held shaky**: deliberately wave the paper. Note the peak `Motion`
     (typically 15-40).

3. **Edit the four constants** in
   [`ml-fastvlm/app/FastVLM App/Iris/IrisConfig.swift`](ml-fastvlm/app/FastVLM%20App/Iris/IrisConfig.swift):

   | Constant | Default | Tune to |
   | --- | --- | --- |
   | `maxMotion` | 4.5 | midway between "held still" and "held shaky" peaks. Lower = wait longer for stability. |
   | `baselineMotion` | 8.0 | minimum value above which the system says "new content". Should be > "held still" peak. |
   | `blurThreshold` | 100.0 | look at `Sharp` values when paper text is legible vs. when it isn't. Set 20% below the legible value. |
   | `cooldownSeconds` | 5.0 | how long the answer stays on screen before the system re-arms. 3 s if you're testing fast, 8 s for a demo. |

4. **Push the change**, wait for TestFlight, repeat. Each round of
   tuning is one commit.

### Cost summary

| Item | Cost |
| --- | --- |
| GitHub Actions on a public repo | $0 |
| Apple Developer Program (you have this) | $99/yr |
| `iris-certs` private repo | $0 (private repos are free) |
| TestFlight | $0 |
| MacInCloud / cloud Mac | $0 (we don't need one) |
| **Total monthly recurring** | **$0 above what you already pay** |
